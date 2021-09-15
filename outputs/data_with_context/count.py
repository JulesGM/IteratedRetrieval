import fire
import matplotlib.pyplot as plt
import transformers

def main(path):
    lens = []
    tokenizer = transformers.AutoTokenizer.from_pretrained("facebook/bart-large")
    with open(path) as f:
        for line in f:
           lens.append(len(tokenizer.encode(line)))

    lens.sort()
    l = len(lens)
    print(f"{lens[int(0.90 * l)]}")
    print(f"{lens[int(0.93 * l)]}")
    print(f"{lens[int(0.95 * l)]}")
    print(f"{lens[int(0.98 * l)]}")
    print(f"{lens[int(0.99 * l)]}")

    # plt.plot(lens)
    # plt.savefig(path + ".png")

if __name__ == "__main__":
    fire.Fire(main)
