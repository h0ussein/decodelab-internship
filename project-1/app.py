def rule_based_chatbot():
    responses = {
        "hello": "Hi there! Welcome to DecodeLabs. How can I assist your engineering track today?",
        "hi": "Hello! Ready to build some deterministic guardrails?",
        "help": "I can assist you with system navigation, project parameters, or basic compliance queries.",
        "status": "All systems operational. Logic engine is running with 100% predictability.",
        "why rules": "Before managing a probabilistic engine, you must master the precision of a logic engine!",
        "about": "I am a White-Box AI system built for absolute safety and zero hallucination risk."
    }

    print("====================================================")
    print("   DecodeLabs Deterministic Guardrail Engine v1.0   ")
    print("      Type 'exit' or 'quit' to terminate loop.      ")
    print("====================================================\n")

    while True:
        raw_input = input("You: ")
        
        clean_input = raw_input.lower().strip()
        
        if clean_input in ['exit', 'quit']:
            print("System: Terminating digital loop. Secure logout successful.")
            break
            
        fallback_msg = "System Warning: Intent unrecognized. Passing to secondary evaluation protocols."
        reply = responses.get(clean_input, fallback_msg)
        
        print(f"Bot: {reply}\n")

if __name__ == "__main__":
    rule_based_chatbot()