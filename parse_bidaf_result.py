from nltk import  word_tokenize

passsage_tokens = word_tokenize("<<<Participants People were eligible for the trial if they were aged at least \
18 years, were in the subacute phase of ischaemic or haemorrhagic stroke (days 5-45 after stroke onset), \
were able to sit unsupported for at least 30 seconds, were considered able to perform aerobic exercise \
by the responsible trial physician, and had a Barthel index score of 65 or less at the time of enrolment. \
The Barthel index measures activities of daily living based on 10 items, with scores ranging from 0 to 100 \
points—higher scores indicating less dependence.15 Key exclusion criteria were intracranial haemorrhage from\
 a ruptured aneurysm or arteriovenous malformation, inability to perform required physical exercise, assisted \
 walking before stroke, or severe cardiac or psychiatric comorbidities. If study requirements were met, trial \
 physicians assessed information on stroke type and medical conditions after written informed consent was obtained. \
 The supplementary file lists the inclusion and exclusion criteria.>>><<< See full tableTable 1. Baseline characteristics \
 of participants stratified by aerobic physical fitness training or relaxation sessions (control group). Values are numbers \
 (percentages) unless stated otherwise>>>")

def make_splitter(passage_tokens: list) -> list:
    # find sequence of splitter_combination and concatinate it
    new_passage_tokens= []
    

    return passsage_tokens


print(make_splitter(passsage_tokens))