from core.llm import Model


def main():
    m = Model(model_name="granite4.1:3b", embeddings_model="all-MiniLM-L6-v2")
    print("FAQ Chatbot — type 'exit' to quit, 'clear' to clear screen\n")

    while True:
        try:
            question = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if question.lower() in ("exit", "quit", "q"):
            break
        if question.lower() == "clear":
            import os
            os.system("cls" if os.name == "nt" else "clear")
            continue
        if not question.strip():
            continue

        result = m.query(question)
        print(f"Bot: {result['response']}\n")


if __name__ == "__main__":
    main()
