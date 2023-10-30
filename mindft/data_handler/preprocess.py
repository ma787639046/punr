from tqdm import tqdm
from collections import defaultdict

def update_dict(dict, key, value=None):
    if key not in dict:
        if value is None:
            dict[key] = len(dict) + 1
        else:
            dict[key] = value

def read_news_bert(news_path, args, tokenizer, mode='train'):
    news = {}
    category_dict = {}
    subcategory_dict = {}
    news_index = {}

    with open(news_path, "r") as f:
        for line in tqdm(f):
            splited = line.strip('\n').split('\t')
            doc_id, category, subcategory, title, _, _, _, _ = splited
            update_dict(news_index, doc_id)

            title = title.lower()
            title_input_ids = tokenizer(title,
                                        max_length=args.num_words_title, 
                                        padding=False,  # do not padding
                                        truncation=True,
                                        return_token_type_ids=False,
                                        return_attention_mask=False,
                                        add_special_tokens=False,
                                        )['input_ids']

            update_dict(news, doc_id, [title_input_ids, category, subcategory])
            if mode == 'train':
                update_dict(category_dict, category)
                update_dict(subcategory_dict, subcategory)

    news_index['unused'] = 0
    # Add "[unused991] [unused992] [unused993]" for pad empty docs
    # pad_docids = [tokenizer.vocab[i] for i in ["[unused991]", "[unused992]", "[unused993]"]]
    pad_docids = tokenizer("Pad doc",
                            max_length=args.num_words_title, 
                            padding=False,  # do not padding
                            truncation=True,
                            return_token_type_ids=False,
                            return_attention_mask=False,
                            add_special_tokens=False,
                            )['input_ids']
    update_dict(news, 'unused', [pad_docids, 'lifestyle', 'lifestyleroyals'])

    # news: dict = {doc_id: [tokenized_title(mlen=30), category, subcategory]}
    # news_index: dict = {doc_id: 1, doc_id: 2 ....}
    # category_dict: dict = {category: 1, category: 2}      # category type id
    # subcategory_dict: dict = {subcategory: 1, subcategory: 2} # subcategory type id
    if mode == 'train':
        return news, news_index, category_dict, subcategory_dict
    elif mode == 'test':
        return news, news_index
    else:
        assert False, 'Wrong mode!'

def get_doc_input_bert(news, news_index, category_dict, subcategory_dict, args):
    news_title = defaultdict(list)
    news_category = defaultdict(int)
    news_subcategory = defaultdict(int)

    for key in tqdm(news):
        title_input_ids, category, subcategory = news[key]
        doc_index = news_index[key]

        news_title[doc_index] = title_input_ids
        news_category[doc_index] = category_dict[category] if category in category_dict else 0
        news_subcategory[doc_index] = subcategory_dict[subcategory] if subcategory in subcategory_dict else 0
        
    return news_title, news_category, news_subcategory