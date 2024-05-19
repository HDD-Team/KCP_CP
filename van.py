from fuzzywuzzy import fuzz
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import scipy.spatial
import numpy as np

model_sim = SentenceTransformer('cointegrated/rubert-tiny2')

df = pd.read_csv(r"classification_cleaned2.csv")
df_fill = pd.read_csv(r"data_to_fill.csv")
df_fill['Схожесть в %'] = None

descriptions = df['Наименование'].tolist()
name_fill = df_fill['name'].tolist()

df[['Код ресурса']].to_csv('resource_codes.csv', index=False, encoding='utf-8')

embeddings_np = np.load('embeddings1.npy')

with open('descriptions.txt', 'w', encoding='utf-8') as f:
    for item in descriptions:
        f.write("%s\n" % item)

df = pd.read_csv('resource_codes.csv')

index = faiss.IndexFlatL2(embeddings_np.shape[1])
index.add(embeddings_np)

def find_closest_matches(input_text, k=5):
    # Correct spelling
    corrected_input_text = input_text

    input_embedding = model_sim.encode([corrected_input_text], convert_to_tensor=True).numpy()

    # Search the FAISS index for the top k closest matches
    D, I = index.search(input_embedding, k=k)

    # Get the closest matches
    matches = []
    for idx in range(k):
        closest_match_index = I[0][idx]
        closest_match = descriptions[closest_match_index]
        resource_code = df.iloc[closest_match_index]['Код ресурса']
        # distance = D[0][idx]
        matches.append((closest_match, resource_code))

    return matches
for input_text in name_fill:
    # input_text = "Арматура стеклопластиковая АСК-4"
    closest_matches = find_closest_matches(input_text, k=100)


    similarity_list = []
    for i, (closest_match, res_code) in enumerate(closest_matches):

        text_embeddings = model_sim.encode([input_text, closest_match])
        similarity = np.squeeze(scipy.spatial.distance.cdist([text_embeddings[0]], [text_embeddings[1]], "cosine"))
        similarity_list.append((similarity, closest_match, res_code))

    similarity_list.sort(key=lambda x: x[0])
    product_names = [similarity_list[0], similarity_list[1]]
    print("Схожесть текстов: {:.2f}%  | ".format(100 * (1 - similarity_list[0][0])), similarity_list[0][1])
    print(similarity_list)

    # Сравнить введенное название с каждым названием товара из базы данных
    similarities = [fuzz.partial_ratio(input_text.lower(), name[1].lower()) for name in product_names]

    # Отсортировать список названий товаров в порядке убывания похожести
    sorted_indices = sorted(range(len(similarities)), key=lambda k: similarities[k], reverse=True)

    # Если похожесть для первого элемента списка меньше заданного порога, то вывести сообщение "Товар не найден"
    answer_name = product_names[sorted_indices[0]]
    if 100 * (1 - answer_name[0]) < 80:
        answer_name = (0, "", "")

    df_fill.loc[df_fill['name'] == input_text, 'КСР наименование'] = answer_name[1]
    df_fill.loc[df_fill['name'] == input_text, 'КСР код'] = answer_name[2]
    df_fill.loc[df_fill['name'] == input_text, 'Схожесть в %'] = 100 * (1 - answer_name[0])
    query = df_fill.loc[df_fill['name'] == input_text, 'КСР наименование']

    print(query)
    print(answer_name)
df.to_csv('filled_csv.csv', index=False)
