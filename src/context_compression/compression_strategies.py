import json
import os
from pathlib import Path
from httpx2 import query
from openai import OpenAI
import tiktoken
import logging
from typing import Any, Dict, List, Optional, Tuple
from mock import _get_mock_search_results
from enum import Enum
from dataclasses import dataclass, field, asdict
from datetime import datetime


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

try:
    from dotenv import load_dotenv
    # 必须显式指向 .env 文件：传目录时 python-dotenv 1.2.x 不会加载任何变量
    load_dotenv(Path(__file__).parent.parent.parent / ".env")
except ImportError:
    pass

# 五种压缩策略
class CompressionStrategy(Enum):
    """Different context compression strategies"""
    # 无压缩
    # 网页原文直接进入上下文
    # 预期：几次工具调用后溢出失败
    # 目的：展示基线问题
    NO_COMPRESSION = "no_compression"
    # 非任务感知：逐页摘要
    # 每页单独 LLM 摘要再拼接
    # 保留页内细节，可能丢跨页关系
    # 多次 LLM 调用；适合来源彼此独立
    NON_CONTEXT_AWARE_INDIVIDUAL = "non_context_aware_individual_summary"
    # 非任务感知：合并摘要
    # 先拼接全部网页再做一次总摘要
    # 更利把握全局，可能丢页级归属
    # 单次 LLM 调用；页多时可能撞限
    NON_CONTEXT_AWARE_COMBINED = "non_context_aware_combined_summary"
    # 上下文感知摘要
    # 结合查询对全部搜索结果做聚焦摘要
    # 相关性更好；多一次 LLM 调用
    CONTEXT_AWARE = "context_aware_summary"
    # 窗口化上下文压缩策略：
    # 最近一次工具调用保留全文，更早历史压缩
    # 细节与效率折中
    # 只压缩尚未标记 [COMPRESSED] 的消息
    WINDOWED_CONTEXT = "windowed_context"
    # 带引用的上下文感知摘要
    # 在 #4 基础上加引用与来源链接
    # 利于追问；上下文略大
    CONTEXT_AWARE_CITATIONS = "context_aware_with_citations"

@dataclass
class CompressedContent:
    """Represents compressed content"""
    original_length: int
    compressed_length: int
    content: str
    citations: List[Dict[str, str]] = field(default_factory=list)
    strategy: CompressionStrategy = CompressionStrategy.NO_COMPRESSION
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """转换为可直接 JSON 序列化的 dict（strategy 枚举转为其 value）"""
        data = asdict(self)
        data["strategy"] = self.strategy.value
        return data


class ContextCompressor:
    """处理不同的上下文压缩策略"""
    def __init__(self,strategy: CompressionStrategy,api_key: str=os.getenv("API_KEY"),enable_streaming: bool=False) -> None:
        self.strategy = strategy
        self.api_key = api_key
        self.enable_streaming = enable_streaming
        self.model = os.getenv("MODEL_NAME")
        self.client = OpenAI(api_key=api_key,base_url=os.getenv("BASE_URL"))
        self.max_tokens = int(os.getenv("MODEL_MAX_TOKENS"))        # tiktoken 编码器
        self.encoding = tiktoken.get_encoding("cl100k_base")

        logging.info(f"使用压缩策略: {strategy.value},streaming: {enable_streaming}")

    # 计算令牌数
    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a string"""
        try:
            return len(self.encoding.encode(text))
        except Exception as e:
            logging.error(f"Error encoding text: {e}")
            return len(text)

    def compress_search_results(self, search_results:Dict[str, Any]) -> CompressedContent:
        """压缩搜索结果基于选中的压缩策略"""
        # 无压缩
        if self.strategy == CompressionStrategy.NO_COMPRESSION:
            return self._no_compression(search_results)
        # 非任务感知：逐页摘要
        elif self.strategy == CompressionStrategy.NON_CONTEXT_AWARE_INDIVIDUAL:
            return self._non_context_aware_individual(search_results)
        # 非任务感知：合并摘要
        elif self.strategy == CompressionStrategy.NON_CONTEXT_AWARE_COMBINED:
            return self._non_context_aware_combined(search_results)
        # 上下文感知摘要
        elif self.strategy == CompressionStrategy.CONTEXT_AWARE:
            return self._context_aware_summary(search_results)
        # 窗口化上下文压缩策略：
        elif self.strategy == CompressionStrategy.WINDOWED_CONTEXT:
            return self.compress_for_history(search_results)
        # 带引用的上下文感知摘要
        elif self.strategy == CompressionStrategy.CONTEXT_AWARE_CITATIONS:
            return self._context_aware_with_citations(search_results)
        else:
            raise ValueError(f"Unknown compression strategy: {self.strategy}")

    def _no_compression(self, search_results:Dict[str, Any]) -> CompressedContent:
        """无压缩策略，直接返回原始搜索结果"""
        all_content = []
        total_length = 0

        for result in search_results.get("results", []):
            content = f"""
                ===== Search Result =====
                Title: {result.get("title", "N/A")}
                URL: {result.get("link", "N/A")}
                Snippet: {result.get("snippet", "N/A")}

                Full context: {result.get("content", "N/A")}
                ========================
            """
            all_content.append(content)
            total_length += len(result.get("content", ""))

        full_content = "\n\n".join(all_content)

        # 返回压缩后的内容
        return CompressedContent(
            original_length=total_length, 
            compressed_length=len(full_content), 
            content=full_content,
            strategy=CompressionStrategy.NO_COMPRESSION,
        )

    # 非任务感知：逐页摘要
    def _non_context_aware_individual(self, search_results:Dict[str, Any]) -> CompressedContent:
        """非任务感知：逐页摘要策略，逐页总结后再合并"""
        summaries = []
        total_original = 0

        for result in search_results.get("results", []):
            if not result.get("content"):
                continue
            original_content = result.get("content","")
            total_original+=len(original_content)

            try:
                prompt = f"""用 2-3 个段落将下面的 web 页面内容总结，请用中文输出：
                    Title: {result.get("title", "N/A")}
                    URL: {result.get("link", "N/A")}
                    Content: {original_content[:5000]}

                    提供简明摘要:
                """
                prompt_tokens = self.count_tokens(prompt)
                logger.info(
                    f"非上下文感知摘要 - Prompt tokens: {prompt_tokens}, Prompt length: {len(prompt)} 字符")
                
                # 使用流式处理
                if self.enable_streaming:
                    print(f"\n📝 总结: {result.get('title', 'N/A')[:50]}...", end=" ", flush=True)
                    stream = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": "你是一个乐于助人的助手，可以创建简洁的摘要。"},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=float(os.getenv("MODEL_TEMPERATURE")),
                        stream=True,
                        max_tokens=int(os.getenv("MODEL_MAX_TOKENS")),
                    )
                    summary_parts = []
                    for chunk in stream:
                        # delta 是 pydantic 对象（ChoiceDelta），不能用 .get()，应直接访问属性
                        if chunk.choices and chunk.choices[0].delta.content:
                            delta_text = chunk.choices[0].delta.content
                            print(delta_text, end="", flush=True)
                            summary_parts.append(delta_text)
                        print(f"chunk: \n {chunk} \n")
                    print("\n")  # New lines after streaming
                    compressed = ''.join(summary_parts)
                else:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": "你是一个乐于助人的助手，可以创建简洁的摘要。"},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=float(os.getenv("MODEL_TEMPERATURE")),
                        max_tokens=int(os.getenv("MODEL_MAX_TOKENS")),
                    )
                    print(f"response: \n {response.model_dump_json()} \n")
                    compressed = response.choices[0].message.content
                return CompressedContent(
                    original_length=len(original_content), 
                    compressed_length=len(compressed), 
                    content=compressed,
                    strategy=CompressionStrategy.NON_CONTEXT_AWARE_INDIVIDUAL,
                )
            except Exception as e:
                logging.error(f"Error summarizing content: {e}")
                # 摘要失败时返回带错误信息的对象，避免隐式返回 None 引发下游 AttributeError
                return CompressedContent(
                    original_length=len(original_content),
                    compressed_length=0,
                    content=f"Error summarizing content: {e}",
                    strategy=CompressionStrategy.NON_CONTEXT_AWARE_INDIVIDUAL,
                )

        # 结果为空或全部缺少 content 时，返回空结果，保证函数永不返回 None
        return CompressedContent(
            original_length=0,
            compressed_length=0,
            content="",
            strategy=CompressionStrategy.NON_CONTEXT_AWARE_INDIVIDUAL,
        )

    # 非任务感知：合并摘要
    def _non_context_aware_combined_summary(self, search_results:Dict[str, Any]) -> CompressedContent:
        """非任务感知：合并摘要策略，先将所有内容合并再做一次摘要"""
        all_content = []
        total_origin = 0
        max_char_per_page = 5000

        for result in search_results.get("results", []):
            if not result.get("content"):
                continue
            original_content = result.get("content","")
            total_origin+=len(original_content)
    
            all_content.append(f"""
                ======= Page: {result.get("title", "N/A")} =======
                URL: {result.get("URL", "N/A")}
                Content: {original_content[:5000]}
            """)
        if not all_content:
            return CompressedContent(
                original_length=0,
                compressed_length=0,
                content="no content available",
                strategy=CompressionStrategy.NON_CONTEXT_AWARE_COMBINED,
            )
        combined_content = "\n\n".join(all_content)

        try:
            prompt = f"""全面总结以下组合网页内容:
                
                {combined_content}

                要求：
                1.为所有页面内容提供一个全面的摘要。
                2.包括来自每个来源的关键信息
                3.保持事实准确性
                4.最大长度：{os.getenv("MODEL_MAX_TOKENS")} tokens

                提供一个完整的总结：
            """
            prompt_tokens = self.count_tokens(prompt)
            logger.info(
                f"非上下文感知摘要 - Prompt tokens: {prompt_tokens}, Prompt length: {len(prompt)} 字符")

            if self.enable_streaming:
                stream = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "你是一个乐于助人的助手，可以创建简洁的摘要。"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=float(os.getenv("MODEL_TEMPERATURE")),
                    max_tokens=int(os.getenv("MODEL_MAX_TOKENS")),
                    stream=True,
                )
                summary_parts = []
                for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        delta_text = chunk.choices[0].delta.content
                        print(f"\ndelta_text: {delta_text}\n", end="", flush=True)
                        summary_parts.append(delta_text)

                    summary = ''.join(summary_parts)
                
            else:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "你是一个乐于助人的助手，可以创建简洁的摘要。"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=float(os.getenv("MODEL_TEMPERATURE")),
                    max_tokens=int(os.getenv("MODEL_MAX_TOKENS")),
                )
                summary = response.choices[0].message.content

                return CompressedContent(
                    original_length=total_origin,
                    compressed_length=len(summary),
                    content=summary,
                    strategy=CompressionStrategy.NON_CONTEXT_AWARE_COMBINED,
                )

        except Exception as e:
            logging.error(f"Error summarizing content: {e}")
            return CompressedContent(
                original_length=total_origin,
                compressed_length=0,
                content=f"Error summarizing content: {e}",
                strategy=CompressionStrategy.NON_CONTEXT_AWARE_COMBINED,
            )

    # 上下文感知摘要
    def _context_aware_summary(self, search_results:Dict[str, Any],query:str,current_context:Optional[str]=None) -> CompressedContent:
        """上下文感知摘要策略，先将搜索结果与当前上下文合并，然后进行摘要"""
        all_content = []
        total_origin = 0
        max_char_per_page = 5000
        for result in search_results.get("results", []):
            if not result.get("content"):
                continue
            original_content = result.get("content","")
            total_origin+=len(original_content)
    
            all_content.append(f"""
                ======= Page: {result.get("title", "N/A")} =======
                URL: {result.get("URL", "N/A")}
                Content: {original_content[:max_char_per_page]}
            """)
        combined_content = "\n\n".join(all_content)

        try:
            prompt = f"""给一个查询问题 '{query}'：
                {f"Current Context: {current_context[:1000]}" if current_context else ""}
                分析以下搜索结果，并提供直接针对查询的重点摘要。
                专注于提取与回答最相关的信息：{query}
                {combined_content}

                要求：
                1. 只关注与查询相关的信息
                2. 优先考虑当前/最近的信息
                3. 包括具体姓名、日期和隶属关系
                4. 最大长度：{os.getenv("MODEL_MAX_TOKENS")} tokens

                提供以查询为重点的摘要：
            """
            prompt_tokens = self.count_tokens(prompt)
            logger.info(
                f"上下文感知摘要 - Prompt tokens: {prompt_tokens}, Prompt length: {len(prompt)} 字符")
            
            if self.enable_streaming:
                stream = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "你是一个乐于助人的助手，可以创建简洁的摘要。"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=float(os.getenv("MODEL_TEMPERATURE")),
                    max_tokens=int(os.getenv("MODEL_MAX_TOKENS")),
                    stream=True,
                )
                for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        delta_text = chunk.choices[0].delta.content
                        print(f"\ndelta_text: {delta_text}\n", end="", flush=True)
                        summary_parts.append(delta_text)
                
                summary = ''.join(summary_parts)
                return CompressedContent(
                    original_length=total_origin,
                    compressed_length=len(summary),
                    content=summary,
                    strategy=CompressionStrategy.CONTEXT_AWARE,
                )
            else:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "你是一个乐于助人的助手，可以创建简洁的摘要。"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=float(os.getenv("MODEL_TEMPERATURE")),
                    max_tokens=int(os.getenv("MODEL_MAX_TOKENS")),
                )
                summary = response.choices[0].message.content

                return CompressedContent(
                    original_length=total_origin,
                    compressed_length=len(summary),
                    content=summary,
                    strategy=CompressionStrategy.CONTEXT_AWARE,
                )
        except Exception as e:
            logging.error(f"Error summarizing content: {e}")
            return CompressedContent(
                original_length=total_origin,
                compressed_length=0,
                content=f"Error summarizing content: {e}",
                strategy=CompressionStrategy.CONTEXT_AWARE,
            )
    # 压缩消息历史记录的内容（用于窗口上下文策略）
    def compress_for_history(
        self,
        content: str,
        tool_name: str,
        query: str,
        preserve_citations: bool = True) -> CompressedContent:
        """压缩消息历史记录的内容（用于窗口上下文策略）"""
        original_length = len(content)

        try:
            prompt = f"""Compress the following {tool_name} results into a concise summary that preserves key information.
                Focus on information relevant to: {query}

                Original content:
                {content[:10000]}

                Requirements:
                1. Keep all important facts, names, dates, and affiliations
                2. Remove redundant information
                3. Maintain clarity and coherence
                {"4. Include [Source: URL] citations for important facts" if preserve_citations else ""}
                5. Maximum length: {os.getenv("MODEL_MAX_TOKENS")} tokens

                Provide a focused summary:
            """

            prompt_tokens = self.count_tokens(prompt)
            logger.info(
                f"Windowed Context Compression - Prompt tokens: {prompt_tokens}, Prompt length: {len(prompt)} 字符")

            if self.enable_streaming:
                stream = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "你是一个乐于助人的助手，可以创建简洁的摘要。"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=float(os.getenv("MODEL_TEMPERATURE")),
                    max_tokens=int(os.getenv("MODEL_MAX_TOKENS")),
                    stream=True,
                )
                summary_parts = []
                for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        delta_text = chunk.choices[0].delta.content
                        print(f"\ndelta_text: {delta_text}\n", end="", flush=True)
                        summary_parts.append(delta_text)

                summary  = ''.join(summary_parts)
                return CompressedContent(
                    original_length=original_length,
                    compressed_length=len(summary),
                    content=summary,
                    strategy=CompressionStrategy.WINDOWED_CONTEXT,
                )
            else:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "你是一个乐于助人的助手，可以创建简洁的摘要。"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=float(os.getenv("MODEL_TEMPERATURE")),
                    max_tokens=int(os.getenv("MODEL_MAX_TOKENS")),
                )
                summary = response.choices[0].message.content
                return CompressedContent(
                    original_length=original_length,
                    compressed_length=len(summary),
                    content=summary,
                    strategy=CompressionStrategy.WINDOWED_CONTEXT,
                )
        except Exception as e:
            logging.error(f"Error compressing history: {e}")
            return CompressedContent(
                original_length=original_length,
                compressed_length=0,
                content=f"Error compressing history: {e}",
                strategy=CompressionStrategy.WINDOWED_CONTEXT,
            )

        return CompressedContent(
            original_length=len(content),
            compressed_length=len(content),
            content=content,
            strategy=CompressionStrategy.WINDOWED_CONTEXT,
        )

    def _context_aware_with_citations(self, search_results: List[Dict[str, Any]], current_context: str, query: str) -> str:
        """压缩搜索结果（用于上下文感知策略）"""
        source = []
        all_content = []
        total_original = 0
        max_chars_per_page = 5000

        for index, result in enumerate(search_results.get("results", [])):
            source_id = f"{index+1}"
            original_content = result.get("content", "")
            total_length += len(original_content)

            source.append({
                "id": source_id,
                "content": original_content[:max_chars_per_page],
                "url": result.get("url", "N/A"),
            })

            all_content.append(f"""
                {source_id} Title: {result.get("title", "N/A")}
                {original_content[:max_chars_per_page]}
            """)
        combined_content = "\n\n".join(all_content)

        try:
            prompt = f"""Given the search query: "{query}"
                {f"Current context: {current_context[:1000]}" if current_context else ""}

                Analyze the following search results and provide a focused summary with citations.

                Search Results (with source IDs):
                {combined_content}

                Requirements:
                1. Focus on information relevant to: {query}
                2. Include inline citations using [1], [2], etc. for each fact
                3. Prioritize current/recent information
                4. Include specific names, dates, and affiliations with citations
                5. Maximum length: {self.max_tokens} tokens

                Provide a query-focused summary with citations:"""

            prompt_tokens = self.count_tokens(prompt)
            logger.info(
                f"Context-aware Compression with Citations - Prompt tokens: {prompt_tokens}, Prompt length: {len(prompt)} 字符")
            
            if self.enable_streaming:
                stream = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "你是一个乐于助人的助手，可以创建简洁的摘要。"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=float(os.getenv("MODEL_TEMPERATURE")),
                    max_tokens=int(os.getenv("MODEL_MAX_TOKENS")),
                    stream=True,
                )
                summary_parts = []
                for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        delta_text = chunk.choices[0].delta.content
                        print(f"\ndelta_text: {delta_text}\n", end="", flush=True)
                        summary_parts.append(delta_text)
                summary = ''.join(summary_parts)
            else:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "你是一个乐于助人的助手，可以创建简洁的摘要。"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=float(os.getenv("MODEL_TEMPERATURE")),
                    max_tokens=int(os.getenv("MODEL_MAX_TOKENS")),
                )
                summary = response.choices[0].message.content

            source_list = "\n\nSources:\n"
            for source in sources:
                source_list += f"- [{source['id']}] {source['title']} - {source['url']}\n"

            final_content = summary + source_list

            return CompressedContent(
                original_length=total_original,
                compressed_length=len(final_content),
                content=final_content,
                strategy=CompressionStrategy.CONTEXT_AWARE_CITATIONS,
            )
        except Exception as e:
            return CompressedContent(
                original_length=total_original,
                compressed_length=0,
                content=f"Error compressing search results: {e}",
                strategy=CompressionStrategy.CONTEXT_AWARE_CITATIONS,
            )
if __name__ == "__main__":
    search_results = _get_mock_search_results(query="Sam Altman current position OpenAI CEO 2025 ")
    compressor = ContextCompressor(CompressionStrategy.NON_CONTEXT_AWARE_INDIVIDUAL, enable_streaming=False)

    compressed_content = compressor.compress_search_results(search_results)
    print(f'Compressed content: \n {json.dumps(compressed_content.to_dict(), indent=4, ensure_ascii=False)}')
