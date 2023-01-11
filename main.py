from utils.loaders.neural_translator_loader import *

if __name__ == "__main__":
    dataset_path = join(config['basedir'],
                        'data/neural_translation_dataset.pickle')
    with open(dataset_path, "rb") as f:
        cond_vecs = pickle.load(f)
        noise_vecs = pickle.load(f)
        sent_vecs = pickle.load(f)

    prompts = get_classes()
    tokenizer = get_tokenizer()
    text_encoder = get_text_encoder()

    prompts = [get_text_embeds(tokenizer, text_encoder, [
                               p])[1].detach().cpu().numpy() for p in prompts]
    print(prompts[0])

    label = []
    for j in range(len(cond_vecs)):
        idx = -1
        has = False
        for i in range(len(prompts)):
            b = (abs(cond_vecs[j] - prompts[i]) < 0.01).all()
            if b:
                idx = i
                label.append(i)
                if has:
                    print("fuck")
                has = True
        if idx == -1:
            print("jeez")

    print(label)

    with open(dataset_path, "wb") as f:
        pickle.dump(label, f, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(noise_vecs, f, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(sent_vecs, f, protocol=pickle.HIGHEST_PROTOCOL)
