from pyweka import main

print("Welcome to Money Guard!")
while(True):
    cmd = input("What do you want to do? Type help for the list of available commands, quit to exit: ")
    if cmd == 'help':
        print('List of commands:\n',
              '- help: gives a detailed list of commands\n',
              '- retrain: requires password. Retrain the classification model with the new transactions\n',
              '- classify: tells you if the transactions is possibly a fraud or not. Requires the transaction as input\n',
              '- stream: given a data stream (simulated by a file), starts an incremental algorithm to classify the transactions\n',
              '- quit: close the application')
    elif cmd == 'retrain':
        main()
    elif cmd == 'classify':
        classifyOne()
    elif cmd == 'stream':
        stream()
    elif cmd == 'quit':
        break
    else:
        print("Sorry, we didn't recognize the command, can you type it correctly? Use help to check the right syntax!")

