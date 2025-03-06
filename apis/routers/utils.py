import bm25s
import pandas as pd
import logging
import os
import re
import io
import numpy as np
import logger
import time


CSV_CACHE="./data/hscode_main_pretty.csv"

logger = logging.getLogger(__name__)

def upper_case(text):
    return text.upper()


def lower_case(text):
    return text.lower()


def check_len_text(text):
    return len(text)


def remove_extra_spaces(text):
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def remove_emails_and_dates(description):
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    description = re.sub(email_pattern, ' ', description)

    date_pattern = r'\b(?:\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}|\d{4}[-/\.]\d{1,2}[-/\.]\d{1,2})\b'
    description = re.sub(date_pattern, ' ', description)

    return description


def delete_not_signficiant_char(text):
    text = text.replace(',', ' ')
    special_chars = r'[~@!`#=\|\+\_\-\$\^&*(){}\[\]"\';:?,\\.]'
    text = re.sub(special_chars, ' ', text)

    return text


def process_sign_percent_and_devide(text):
    text = re.sub(r'(%{2,}|/{2,})', ' ', text)
    return text


def process_sign_less_more(text):
    text = re.sub(r'(?<!<)<(?!<)', ' less than ', text)
    text = re.sub(r'(?<!>)>(?!>)', ' more than ', text)

    text = re.sub(r'<{2,}|>{2,}', ' ', text)
    return text


def main_function_process(text):
    try:
        text = upper_case(text)
        text = remove_emails_and_dates(text)
        text = delete_not_signficiant_char(text)
        text = process_sign_percent_and_devide(text)
        text = process_sign_less_more(text)
        text = lower_case(text)
        text = remove_extra_spaces(text)

    except Exception as e:
        logging.error(f"Error when processing: {e}")
        return None

    return text


def process_docs(docs, scores):
    new_docs = []
    new_scores = []
    new_docs_2 = []
    new_scores_2 =[]
    n = len(docs[0])
    for i in range(n):
        score = scores[0][i]
        if int(score) != 0:
            new_docs.append(docs[0][i])
            new_scores.append(scores[0][i])
    new_docs_2.append(new_docs)
    new_scores_2.append(new_scores)
    return new_docs_2, new_scores_2


def cal_score(scores):
    start_time = time.time()
    try:
        scores = np.array(scores, dtype=float)
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

        mean_score = np.mean(scores)
        std_dev = np.std(scores)
        if std_dev == 0:
            logger.warning("Standard deviation is zero, setting z-scores to zero.")
            z_scores = np.zeros_like(scores)
        else:
            z_scores = (scores - mean_score) / std_dev

        confidence_levels = (1 + (1 / (1 + np.exp(-z_scores)))) / 2 * 100
        logger.info("Confidence levels calculated successfully.")
        print("Confidence score takes: ",(time.time() - start_time))
        confidence_levels = [float(x) for x in confidence_levels.flatten().tolist()]
        return confidence_levels

    except Exception as e:
        logger.error(f"Failed to calculate confidence levels: {e}")
        return None


def mapping_hscode(hs_codes):
    start_time = time.time()
    try:
        df_hscode = pd.read_csv(CSV_CACHE, dtype={"hs_code_6": "object",'hs_code_8':'object'})
        # print(df_hscode)
        logger.info("Loaded hscode_main_pretty.csv successfully.")
        print("Loaded hscode_main_pretty.csv takes: ",(time.time() - start_time))
    except Exception as e:
        logger.error(f"Error while loading hscode_main_pretty.csv: {e}")
        return [], [], [], []

    descriptions, chapters, headings, subheadings, full_descriptions = [], [], [], [], []

    try:
        for hs_code in hs_codes:
            description_row = df_hscode.loc[df_hscode['hs_code_8'] == hs_code, 'full_description']
            descriptions.append(description_row.iloc[0] if not description_row.empty else "")
        logger.info("Descriptions mapped successfully.")
        print("Descriptions mapped takes: ",(time.time() - start_time))
    except Exception as e:
        logger.error(f"Error mapping descriptions: {e}")

    try:
        # print("HS OCDES",hs_codes)
        for hs_code in hs_codes:
            chapter_row = df_hscode.loc[df_hscode['hs_code_8'] == hs_code, 'chapter']
            heading_row = df_hscode.loc[df_hscode['hs_code_8'] == hs_code, 'heading']
            subheading_row = df_hscode.loc[df_hscode['hs_code_8'] == hs_code, 'subheading']
            fulldescription_row = df_hscode.loc[df_hscode['hs_code_8'] == hs_code, 'full_description']
            # print("Chap",chapter_row)
            # print("Head",heading_row)
            # print("Sub",subheading_row)
            # print("Full",fulldescription_row)
            chapters.append(chapter_row.iloc[0] if not chapter_row.empty else "")
            headings.append(heading_row.iloc[0] if not heading_row.empty else "")
            subheadings.append(subheading_row.iloc[0] if not subheading_row.empty else "")
            full_descriptions.append(fulldescription_row.iloc[0] if not fulldescription_row.empty else "")
        # print(headings)
        # print(subheadings)
        # print(full_descriptions)
        logger.info("Chapter, heading, subheading and full_description mapped successfully.")
        print("Chapter, heading, and subheading mapped takes: ",(time.time() - start_time))
    except Exception as e:
        logger.error(f"Error mapping chapter, heading, subheading, full_descriptions: {e}")

    # return descriptions, chapters, headings, subheadings
    return descriptions, chapters, headings,subheadings, full_descriptions



async def format_result(docs, confidence_levels):

    if any(np.isnan(confidence_levels) | np.isinf(confidence_levels)):
        logger.warning("Confidence levels contain NaN or infinite values, replacing with 0.")
        confidence_levels = np.nan_to_num(confidence_levels, nan=0.0, posinf=0.0, neginf=0.0)

    hs_codes = []
    print("docs: ", docs)
    try:
        for doc in docs:
            for d in doc:
                text = d['text']
                if 'hs_code: ' in text:
                    hs_codes.append(text.split('hs_code: ')[1].split()[0])
                else:
                    logger.warning(f"'hs_code: ' not found in text: {text}")
        logger.info("hs_codes processed successfully.")
    except Exception as e:
        logger.error(f"Error processing hs_codes from docs: {e}")

    confidences = [np.round(conf, 2) for conf in confidence_levels]
    confidences = [float(x) for x in confidences]
    descriptions, chapters,  headings, subheadings, full_descriptions = mapping_hscode(hs_codes)
  
    detail_description = []
    try:
        for index, hs_code in enumerate(hs_codes):
            detail_description.append({
                "chapter": {
                    "code": hs_code[0:2],
                    "description": chapters[index],
                    "chapter_name": f"Chapter {hs_code[0:2]}"
                },
                "heading": {
                    "code": hs_code[2:4],
                    "description": headings[index],
                    "heading_name": f"Heading {hs_code[0:4]}"
                },
                "subheading": {
                    "code": hs_code[4:6],
                    "description": subheadings[index],
                    "subheading_name": f"Subheading {hs_code[0:6]}"
                }
            })
        logger.info("Detailed description formatted successfully.")
    except Exception as e:
        logger.error(f"Error during detailed description formatting: {e}")

    try:
        confidence_levels = sorted(confidence_levels, reverse=True)
        print("Log confi",confidence_levels)
    
        formatted_df = pd.DataFrame({
            'hs_code': hs_codes,
            'confidence': confidence_levels,
            'description': full_descriptions,
            'detail_description': detail_description
        }) 
        formatted_df["confidence"]=confidence_levels
        formatted_df['confidence']=formatted_df['confidence'].apply(lambda x:round(x,2))

    except Exception as e:
        print("Error when sort score: ",e)
    
    formatted_df = formatted_df.drop_duplicates(subset=['hs_code'])
    print("Log df_formatted after sort",formatted_df)

    return formatted_df.to_dict(orient='records')




async def predict_hscode(input_data, retrieval_model):
    sentence = main_function_process(input_data.sentence)
    try:
        tokenized_description = bm25s.tokenize(sentence)
        docs, scores = retrieval_model.retrieve(tokenized_description, k=20)
        docs, scores = process_docs(docs, scores)

        confidence_levels = cal_score(scores)
        results = await format_result(docs,confidence_levels)

    except Exception as e:
        print(f"Error at predict sentence: {e}")
        results = []
    return results

