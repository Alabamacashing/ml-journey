def calculator_tool(expression):
    return eval(expression)

def word_counter_tool(text):
    return len(text.split())

def simple_agent(user_input):
    if any(char.isdigit() for char in user_input) and any(op in user_input for op in ["+", "-", "*", "/"]):
        print("Thought: This looks like a math question. Using calculator tool.")
        result = calculator_tool(user_input)
        print(f"Observation: {result}")
        return result
    else:
        print("Thought: This looks like a text question. Using word counter tool.")
        result = word_counter_tool(user_input)
        print(f"Observation: {result} words")
        return result


simple_agent("15 * 23")
simple_agent("The quick brown fox jumps over the lazy dog")