import argparse

def parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--num', dest='num',
                        help='Number of input wanted, default is 5',
                        default=5, type=int)

    return parser.parse_args()

def main(num):
    print("[MAIN]\tNumber of input selected is {}" .format(num))

    numbers = []
    i = 0

    while(i < num):
        i_enter = input("[MAIN]\tPlease enter {} number... " .format(i))
 
        try:
            val = int(i_enter)
            print("[MAIN]\tInput is an integer number. Number = ", val)
            numbers.append(val)
        except ValueError:
            try:
                val = float(i_enter)
                print("[MAIN]\tInput is a float  number. Number = ", val)
                continue
            except ValueError:
                print("[MAIN]\tNo.. input is not a number. It's a string")
                continue

        i+=1

    numbers.sort()
    print("[MAIN]\tYou stored numbers {}" .format(numbers))

if __name__ == "__main__":
    args = parser()
    main(args.num)