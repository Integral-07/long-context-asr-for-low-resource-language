
"""
This script extracts transcriptions for the "Tuytah Collection by ASAI Take"
from html files downloaded from http://www.aa.tufs.ac.jp/~mmine/kiki_gen/murasaki/.

Anaconda environment used: speech_recognition

The procedure is as follows:
(for each HTML file)
- parse the file with BeautifulSoup
(for each row in the table containing links to WAV files and transcriptions)
- check if a new WAV file starts from this row (some WAV files span multiple
  rows of transcribed text)
- if yes, save the transcription for the previous file and reset the variable
  storing text -> initialize it with the WAV file name and separator (space)
- otherwise, append the transcription to previous rows corresponding to the
  same WAV file

TODO:
OK?? - decide what to do with items like "ne(e)", "henne(h)", "(makapahci)" (at04aj)
- convert all parts in Japanese to katakana/romaji
OK? - handle cases such as "アノーponiwne" (split the part in kana and romaji)

"""

import os
from bs4 import BeautifulSoup as bs
import re
import pykakasi # for transliterating Japanese script to katakana
kks = pykakasi.kakasi()

INPUT_PATH = 'C:/Users/karol/Documents/Ainu & NLP (Kitami)/Voice recordings & textual data/Ainu dictionaries & other language resources/Tuytah Collection (oral literature of the Sakhalin Ainu) by ASAI Take/html/'
INDEX_FILENAME = 'Asai-Murasaki-index'
WAV_PATH = 'C:/Users/karol/Documents/Ainu & NLP (Kitami)/Voice recordings & textual data/Ainu dictionaries & other language resources/Tuytah Collection (oral literature of the Sakhalin Ainu) by ASAI Take/audio/wav (from website)/'

CORRECTIONS = {'X・・・・': "はい、どうぞ。ここからですね。",
			   '7': "'", # At43-89
			   '‘': "'", # normalization
			   '’': "'"} # normalization
METADATA = ["(M)", "M)", "M）", "Ｍ）", "Ｍ)", "(Si)", "Si）", "Si)", "Sa)",
			"テープ切れる", "テープA面切れる", "（咳）", "(咳)", "咳", "（拍手)",
			"（笑声）", "平仮名", "（笑い声M）", "笑。M)", "(井戸)", "M)s",
			"(笑い声)", "（時計の音）",
			"(テープ切れる。少しブランク)",
			"「自分で食ったんだべ",
			"(以下はウンカヨの家でのことを繰り返している)"]
PUNCTUATION = [".", ",", "-", '"', "“", "”", "?", "!", "、", "。", "„",
			   "？", "(", ")", "（", "）", "「", "」", "<", ">", "＜", "＞",
			   "・", "，", "！"]

def main():
	html_files = find_files(INPUT_PATH, "html")
	del html_files[INDEX_FILENAME] # no need to process the index file
	wav_files = find_files(WAV_PATH, "wav")
	for file in html_files:
		soup = parse_html(html_files[file])
		output = []
		text = None
		for row in soup.find_all("tr"):
			ahref = row.select_one('a[href^="http://www.aa.tufs.ac.jp/~mmine/kiki_gen/murasaki/"]')
			if ahref:
				wav_url = ahref['href']
				wav_filename = wav_url.split('/')[-1].replace('.wav', '')
				if text:
					output.append(text)
				if wav_filename in wav_files: # check if WAV file exists
					text = wav_filename
				else:
					text = None
					print("This file does not exist: ", wav_filename, ".wav")
			if text:
				transcr = row.find('td', valign='top').get_text().strip()
				transcr = correct(transcr)
				transcr = clean(transcr)
				transcr = normalize(transcr)
				text = text + ' ' + transcr
		if text:
			output.append(text)
		outfilename = file + ".trans.txt"
		outfilename = outfilename.replace("aj", "") # to make the filenames match WAV filenames
		write_to_file(output, outfilename)

def correct(transcription):
	for corr in CORRECTIONS:
		transcription = transcription.replace(corr, CORRECTIONS[corr])
	return transcription
	
def normalize(transcription):
	"""
	Normalizes the transcribed text by doing the follwing:
		- convert all characters to lower case
		- converts all parts in Japanese to katakana
		- split tokens mixing latin script with Japanese
	"""
	transcription = transcription.lower()
	transcription = conv_jpn_to_kana(transcription)
	transcription = separate_latin_from_jpn(transcription)
	return transcription

def conv_jpn_to_kana(text):
	converted = kks.convert(text)
	kana_tokens = (tok['kana'] for tok in converted)
	return "".join(kana_tokens)

def separate_latin_from_jpn(text):
	"""
	Separates latin script from text in Japanese.
	E.g. "'utahデナイ" -> "'utah デナイ"
	"""
	text = re.sub(r'([^\Wa-zA-Z_0-9])([a-zA-Z_0-9])', r'\1 \2', text)
	text = re.sub(r'([a-zA-Z_0-9])([^\Wa-zA-Z_0-9])', r'\1 \2', text)
	return text

def clean(transcription):
	"""
	Clean the transcribed text by removing irrelevant information:
		- delete metadata, such as 'M' (denoting Murasaki's speech), '(咳)',
		  '（拍手)', '(テープA面切れる)'
		- delete non-phonemic symbols (punctuation, brackets, etc.)
		- remove duplicate spaces and line breaks
	"""
	transcription = remove_metadata(transcription)
	transcription = remove_punc(transcription)
	transcription = remove_multi_spcs(transcription)
	return transcription

def remove_metadata(text):
	"""
	Replaces (pre-specified) metadata with spaces
	(produces excess spaces, so 'remove_multi_spcs'
	must be called after this).
	"""
	for pattern in METADATA:
		text = text.replace(pattern, " ")
	return text

def remove_punc(text):
	"""
	1. Removes brackets found inside a token (as in "ne(e)").
	2. Replaces all other punctuation marks with spaces
	(thus producing excess spaces, so 'remove_multi_spcs'
	must be called after this).
	"""
	# 1
	text = re.sub(r'([a-zA-Z])(\(|\))([a-zA-Z])', r'\1\3', text)
	# 2
	for pattern in PUNCTUATION:
		text = text.replace(pattern, " ")
	return text
	
def remove_multi_spcs(text):
	"""
	Replaces multiple consecutive spaces characters with a single space.
	"""
	text = " ".join(text.split())
	return text

def parse_html(html_doc):
    encoding = 'shift_jis'
    with open(html_doc, 'r', encoding=encoding) as fh:
        return bs(fh)

def find_files(path, extension):
    file_list = [f for f in os.listdir(path) if f.endswith('.'+extension)]
    files = {}
    for file in file_list:
        name = os.path.splitext(file)[0]
        files[name] = path + "/" + file
    return files

def read_lines(filename):
	with open(filename, 'r', encoding='utf-8') as file:
		# return [line.strip() for line in file]
		return [line.strip('\r\n') for line in file]

def write_to_file(output, filename):
	with open(filename, 'w', encoding='utf-8') as out_file:
		for line in output:
			out_file.write(line + "\n")

main()