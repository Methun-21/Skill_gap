import spacy

nlp = spacy.load("en_core_web_sm")

# Extracts named entities from the text.
# Question 4.1: Extract Named Entities
def extract_entities(text):
    """
    Extracts named entities from text using spaCy's NER.
    Focuses on ORG, PERSON, GPE, PRODUCT.
    """
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        if ent.label_ in ["ORG", "PERSON", "GPE", "PRODUCT"]:
            entities.append((ent.text, ent.label_))
    return entities

# Question 4.2: Annotate Training Data
# Manually annotated sentences with skill entities.

# Character counting:
# "Python developer with 5 years of experience"
#  P y t h o n
#  0 1 2 3 4 5
# (0, 6)

# "Expert in Machine Learning and Data Science"
#            M a c h i n e   L e a r n i n g
#            10                          25
# (10, 26)
#                                D a t a   S c i e n c e
#                                31                  42
# (31, 43)

# "Proficient in TensorFlow and PyTorch frameworks"
#                T e n s o r F l o w
#                15              24
# (15, 25)
#                               P y T o r c h
#                               30          36
# (30, 37)

# "Strong SQL and MongoDB database skills"
#        S Q L
#        7   9
# (7, 10)
#               M o n g o D B
#               15          21
# (15, 22)

# "Excellent communication and leadership abilities"
#            c o m m u n i c a t i o n
#            10                      22
# (10, 23)
#                              l e a d e r s h i p
#                              28              37
# (28, 38)

TRAIN_DATA = [
    ("Python developer with 5 years of experience", 
     {"entities": [(0, 6, "SKILL")]}),
    
    ("Expert in Machine Learning and Data Science",
     {"entities": [(10, 26, "SKILL"), (31, 43, "SKILL")]}),
    
    ("Proficient in TensorFlow and PyTorch frameworks",
     {"entities": [(15, 25, "SKILL"), (30, 37, "SKILL")]}),
    
    ("Strong SQL and MongoDB database skills",
     {"entities": [(7, 10, "SKILL"), (15, 22, "SKILL")]}),
    
    ("Excellent communication and leadership abilities",
     {"entities": [(10, 23, "SKILL"), (28, 38, "SKILL")]})
]

# --- Testing the functions ---

# Test for extract_entities
text_input = "John worked at Google and Microsoft in New York. He used TensorFlow and Python."
entities_output = extract_entities(text_input)
print("Extracted Entities:")
print(entities_output)

print("\n" + "="*20 + "\n")

# Print the annotated training data
print("Annotated Training Data:")
for data in TRAIN_DATA:
    print(data)
