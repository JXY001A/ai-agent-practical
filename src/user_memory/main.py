from user_memory.memory_manager import create_memory_manager
from config import Config, MemoryMode
from user_memory.conversation_agent import ConversationAgent
from user_memory.background_memory_processor import BackgroundMemoryProcessor

def start(arg):
    user_id = "jxy001c"
    memory_mode = MemoryMode.JSON_CARD

    agent = ConversationAgent(user_id, memory_mode=memory_mode)

    background_memory_processor = BackgroundMemoryProcessor(user_id=user_id, mode=memory_mode)

    intro_messages = [
        # "Hi! I'm Alex, a software developer who loves Python and machine learning.",
        # "I'm currently working on a recommendation system project using PyTorch.",
        # "I prefer dark themes in my IDE and always use type hints in my Python code."
        "告诉我你知道的有关于我的信息"
    ]

    for i, message in enumerate(intro_messages, 1):
        print(f"\n[会话轮数 {i}]")
        print(f"用户消息: {message}")
        response = agent.chat(message)
        print(f"🤖 Assistant: {response[:50]}..." if len(response) > 50 else f"🤖 Assistant: {response}")

        background_memory_processor.increment_conversation_count()

        results = background_memory_processor.process_recent_conversations()

        operations = results.get('operations', [])
        if operations:
            print("\n 记忆操作:")
            for j, op in enumerate(operations, 1):
                icon = {'add': '➕', 'update': '📝', 'delete': '🗑️'}.get(op['action'], '❓')
                print(f"  {j}. {icon} {op['action'].upper()}: {op.get('content', '')[:80]}...")
        else:
            print("  ℹ️ 没有记忆被添加")

        summary = results.get('summary', {})
        if any(summary.values()):
            print(f"  Summary: {summary.get('add', 0)} added, {summary.get('update', 0)} updated")

    print("\n" + "=" * 40)
    print("💾 记忆状态在会话 1 之后")
    print("=" * 40)
    memory_manager = create_memory_manager(user_id, memory_mode)
    memory_str = memory_manager.get_context_string()
    print(memory_str)

    print("\n" + "=" * 60)
    print("SESSION 2: 记忆调用和更新")
    print("=" * 60)

    agent.reset_session()

    print("🔄 Started new conversation session\n")

    recall_messages = [
        "What do you remember about my work and preferences?",
        "Actually, I recently switched from PyTorch to JAX for better performance.",
        "Can you recommend tools for my recommendation system based on what you know about me?"
    ]

    for i, message in enumerate(recall_messages, 1):
        print(f"\n[会话轮数 {i}]")
        print(f"用户消息: {message}")
        response = agent.chat(message)
        print(f"🤖 Assistant: {response[:50]}..." if len(response) > 50 else f"🤖 Assistant: {response}")
        background_memory_processor.increment_conversation_count()

        print(f"\n📝 Processing memory after conversation {i}...")
        results = background_memory_processor.process_recent_conversations()

    operations = results.get('operations', [])
    if operations:
        print("\n 记忆操作:")
        for j, op in enumerate(operations, 1):
            icon = {'add': '➕', 'update': '📝', 'delete': '🗑️'}.get(op['action'], '❓')
            content = op.get('content', op.get('memory_id', 'N/A'))
            print(f"  {j}. {icon} {op['action'].upper()}: {content[:80]}...")
            if op.get('reason'):
                print(f"     Reason: {op['reason'][:80]}...")
    else:
        print("  ℹ️ 没有记忆需要更新")

    summary = results.get('summary', {})
    if any(summary.values()):
        print(f"  Summary: {summary.get('add', 0)} added, {summary.get('update', 0)} updated")

if __name__ == "__main__":
    start(None)
