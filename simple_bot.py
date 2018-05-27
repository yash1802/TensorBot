import framework

try:
    while True:
        print("Bot: ", framework.response(input("Me: ")))
except EOFError:
    pass
