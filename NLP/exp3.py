def dfa_even_ones(string):
    state = 'q0'  # Start in q0 (even number of 1s)

    for char in string:
        if state == 'q0':
            if char == '1':
                state = 'q1'
            elif char == '0':
                state = 'q0'
            else:
                return False  # invalid input
        elif state == 'q1':
            if char == '1':
                state = 'q0'
            elif char == '0':
                state = 'q1'
            else:
                return False  # invalid input

    return state == 'q0'  # Accept if even number of 1s


# User input
user_input = input("Enter a binary string (0s and 1s only): ")

# Result
if dfa_even_ones(user_input):
    print(f"✅ The string '{user_input}' is ACCEPTED (even number of 1's).")
else:
    print(f"❌ The string '{user_input}' is REJECTED (odd number of 1's).")
