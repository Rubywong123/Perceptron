wget https://nlp.stanford.edu/software/stanford-parser-4.2.0.zip
unzip stanford-parser-4.2.0.zip
wget https://downloads.cs.stanford.edu/nlp/software/stanford-corenlp-4.2.0-models-english.jar
mv stanford-corenlp-4.2.0-models-english.jar stanford-parser-4.2.0
mv stanford-parser-4.2.0 parser
python main.py