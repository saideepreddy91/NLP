import io
import pickle

class RenameUnpickler(pickle.Unpickler):
    """
    Load pickle with custom module path.
    Adapted from: https://stackoverflow.com/a/53327348/6666165

    Note 1: 
    ------
    When you save to pickle file, it basically remembers the module path of dependent objects in that environment.
    E.g., if you have `my_package.my_module` the time you save, 
    when it loads, it will find my_module exactly at `my_package.my_module`, regardless of where you put the my_module!

    This can be a problem if you use the pickle file in another project, where `my_package` doesn't exist. It's still looking for it.

    TL;DR: pickle is sensitive (at load-time) to the environment at save-time. This class handles that (nicely).
    """
    def find_class(self, module, name):
        renamed_module = module
        if module == "text_preprocessor":
            renamed_module = "analysis.text_preprocessor"

        return super(RenameUnpickler, self).find_class(renamed_module, name)

def renamed_load(file_obj):
    return RenameUnpickler(file_obj).load()

def rgba(r, g, b, a=None):
    if a is None:
        return f'rgb({r},{g},{b})'
    else:
        return f'rgba({r},{g},{b},{a})'

def map_to_new_low_and_high(value, from_low, from_high, new_low, new_high):
    if from_low == from_high:
        return new_high
    ratio = (value - from_low)/(from_high - from_low)
    return ratio * (new_high - new_low) + new_low

def get_relative_strengths(values, new_low, new_high):
    """
    Given a list of values, get relative strength (linearly) for each value,
    where minimum value will be 'new_low' and maximum value will be 'new_high'.,
    """    
    assert new_low < new_high
    from_low = min(values)
    from_high = max(values)
    return [map_to_new_low_and_high(v, from_low, from_high, new_low, new_high) for v in values]

def read_sentiment(tarfname):
    """
    Read complete sentiment raw documents, without any preprocessing.
    
    Available fields: ['train_data', 'train_labels', 'dev_data', 'dev_labels', 'le', 'target_labels', 'trainy', 'devy']
    """
    import tarfile
    tar = tarfile.open(tarfname, "r:gz")
    trainname = "train.tsv"
    devname = "dev.tsv"
    for member in tar.getmembers():
        if 'train.tsv' in member.name:
            trainname = member.name
        elif 'dev.tsv' in member.name:
            devname = member.name
            
    def read_tsv(tar, fname):
        member = tar.getmember(fname)
        tf = tar.extractfile(member)
        data = []
        labels = []
        for line in tf:
            line = line.decode("utf-8")
            (label,text) = line.strip().split("\t")
            labels.append(label)
            data.append(text)
        return data, labels
            
    class Data: pass
    sentiment = Data()
    sentiment.train_data, sentiment.train_labels = read_tsv(tar,trainname)
    sentiment.dev_data, sentiment.dev_labels = read_tsv(tar, devname)
    
    from sklearn import preprocessing
    sentiment.le = preprocessing.LabelEncoder()
    sentiment.le.fit(sentiment.train_labels)
    sentiment.target_labels = sentiment.le.classes_
    sentiment.trainy = sentiment.le.transform(sentiment.train_labels)
    sentiment.devy = sentiment.le.transform(sentiment.dev_labels)
    tar.close()
    
    return sentiment

def hex_string_to_rgb(hex_string):
    h = hex_string.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def read_news_data(tarfname='news_dataset.tar.gz'):
    import tarfile
    tar = tarfile.open(tarfname, "r:gz")
    name_train_data = "news_dataset/news_data_train.txt"
    name_train_labels = "news_dataset/news_labels_train.txt"
    name_test_data = "news_dataset/news_data_test.txt"
    name_test_labels = "news_dataset/news_labels_test.txt"
    
    def read_lines_from_tar(fname):
        member = tar.getmember(fname)
        tf = tar.extractfile(member)
        return [eval(line.decode('utf-8')).strip() for line in tf]
    
    class Data: pass
    news = Data()
    news.train_data, news.train_labels = read_lines_from_tar(name_train_data), read_lines_from_tar(name_train_labels)
    news.test_data, news.test_labels = read_lines_from_tar(name_test_data), read_lines_from_tar(name_test_labels)
    tar.close()
    return news  
