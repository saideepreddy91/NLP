from analysis.apos import apos 
import string

apos = {k.lower():v.lower() for k,v in apos.items()}

class TextPreprocessor:
    """
    Text preprocessing used in TfidfVectorizer.
    """
    def __init__(self, 
               expand_apostrophe=True,
              apos=apos):

        assert not expand_apostrophe or apos is not None, "Must provide apos (map from contraction -> expansion) set."

        self.expand_apostrophe = expand_apostrophe

        # if not expand
        self.remove_punctuation = str.maketrans('','',string.punctuation)

        # if expand
        self.remove_punctuation_EXCEPT_apostrophe = str.maketrans('','',string.punctuation.replace("'", ""))
        self.remove_apos = str.maketrans('','',"'")


    def preprocess(self, s):
        assert isinstance(s, str), 's must be string'

        s = s.lower()

        if self.expand_apostrophe:
            s = s.translate(self.remove_punctuation_EXCEPT_apostrophe)
            words = [w for w in s.split()]
            words = [apos[word] if word in apos else word for word in words]
            s = ' '.join(words)
            
            # clean up the rest of non-matching apos
            s = s.translate(self.remove_apos)
        else:
            s = s.translate(self.remove_punctuation)

            # ensure one space
            s = ' '.join(s.split())
        return s