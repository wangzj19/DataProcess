# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import sys

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    lr = 0
    cur_epoch = 17
    base_lr =0.001
    lr_schedule = [4, 8, 12, 14, 16]
    lr_schedule.append(sys.maxsize)
    for i, e in enumerate(lr_schedule):
        if cur_epoch < e:
            lr = base_lr * (0.1 ** i)
            break
    if lr == 0:
        lr = base_lr

    print(f"now lr:{lr}")
    print( lr)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
