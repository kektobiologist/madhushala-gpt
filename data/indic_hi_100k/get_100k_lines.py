# get the first 100k lines form hi.txt and put them in hi_100k.txt

import sys
import os

def main():
    with open(os.path.join(os.path.dirname(__file__), 'hi.txt'), 'r') as f:
        with open(os.path.join(os.path.dirname(__file__), 'hi_100k.txt'), "w") as g:
            for i in range(100000):
                line = f.readline()
                if i < 10:
                    print(line)
                g.write(line)
            g.close()
        f.close()

if __name__ == "__main__":
    main()