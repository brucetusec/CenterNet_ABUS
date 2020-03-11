import os, random

def main():    
    with open(root + 'annotations/old_all.txt', 'r') as f:
        lines = f.read().splitlines()
        random.shuffle(lines)
    
    with open(root + 'annotations/rand_all.txt', 'w') as f:
        new_lines = []
        for li in lines:
            new_lines.append(li +'\n')
            
        f.writelines(new_lines)

if __name__ == '__main__':
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'data/sys_ucc/')
    main()