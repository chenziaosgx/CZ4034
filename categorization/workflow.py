import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def getDataFromSolar():
    pass

def updateSolar():
    pass

def getKeywordsFromSolar():
    pass

def removeURI(txt):
    import re
    text = re.sub(r'^https?:\/\/.*[\r\n]*', '', txt, flags=re.MULTILINE)
    return text

def extract_candidate_words(text, good_tags=set(['JJ','JJR','JJS','NN','NNP','NNS','NNPS'])):
    import itertools, nltk, string

    # exclude candidates that are stop words or entirely punctuation
    punct = set(string.punctuation)
    stop_words = set(nltk.corpus.stopwords.words('english'))
    # tokenize and POS-tag words
    tagged_words = itertools.chain.from_iterable(nltk.pos_tag_sents(nltk.word_tokenize(sent)
                                                                    for sent in nltk.sent_tokenize(text)))
    # filter on certain POS tags and lowercase all words
    word_n_tages =[]
    for word, tage in tagged_words:
        word_n_tages.append([word, tage])

    candidates = [[word.lower(), tag]for word, tag in tagged_words
                  if tag in good_tags and word.lower() not in stop_words
                  and not all(char in punct for char in word)]

    return candidates

def extract_chunks_regular_expression(text, grammar=r'KT: {(<JJ>* <NN.*>+ <IN>)? <JJ>* <NN.*>+}', len_limit = 100):
    import itertools, nltk, string

    # exclude candidates that are stop words or entirely punctuation
    punct = set(string.punctuation)
    stop_words = set(nltk.corpus.stopwords.words('english'))
    # tokenize, POS-tag, and chunk using regular expressions
    chunker = nltk.chunk.regexp.RegexpParser(grammar)
    tagged_sents = nltk.pos_tag_sents(nltk.word_tokenize(sent) for sent in nltk.sent_tokenize(text))
    all_chunks = list(itertools.chain.from_iterable(nltk.chunk.tree2conlltags(chunker.parse(tagged_sent))
                                                    for tagged_sent in tagged_sents))
    # join constituent chunk words into a single chunked phrase
    cands = []

    for key, group in itertools.groupby(all_chunks, lambda (word, pos, chunk): chunk != 'O'):
        if key:
            word_ = ""
            pos_ = ""
            len_ = 0
            for word, pos, chunk in group:
                word_ = word_ + ' ' + word
                pos_ = pos_ + ' ' + pos
                len_ +=1

            if word_ not in stop_words and not all(char in punct for char in word_) and len_ < len_limit :

                cands.append([pos_.lstrip(' '),word_.lstrip(' '), str(len_) +' gram' ])

    res = pd.DataFrame.from_records(cands, columns=['tag', 'word', 'expressionCount'])

    return res

def getWordPattern(text):
    import itertools, nltk, string
    stop_words = set(nltk.corpus.stopwords.words('english'))

    punct = set(string.punctuation)
    tagged_words = itertools.chain.from_iterable(nltk.pos_tag_sents(nltk.word_tokenize(sent)
                                                                    for sent in nltk.sent_tokenize(text)))

    # candidates = [[ tag, word.lower()] for word, tag in tagged_words
    #               if not all(char in punct for char in word)]
    candidates = [[tag, word.lower()] for word, tag in tagged_words
                  if not all(char in punct for char in word) and word.lower() not in stop_words]

    res = pd.DataFrame.from_records(candidates, columns=['tag', 'word'])

    return res

def processWords(multi, single):
    from nltk.stem import WordNetLemmatizer
    # stem the word and bring back to normal form
    wordnet_lemmatizer = WordNetLemmatizer()
  #  multi_stermmer = [stermmer.stem(i) for i in multi]
    multi_stermmer = []
    for i in multi:
        wds = i.split()
        wds_ = ' '.join(wordnet_lemmatizer.lemmatize(u) for u in wds)
        # for w in wds:
        #     wds_.append(stermmer.stem(w))
        multi_stermmer.append(wds_)


    single_stermmer = [wordnet_lemmatizer.lemmatize(u.lower()) for u in single]
 #   final_stermmer = multi_stermmer + single_stermmer
    final_result = []
    unique_single_stermmer = []
    for u in xrange(len(single_stermmer)):
        if single_stermmer[u].lower() not in unique_single_stermmer:
            unique_single_stermmer.append(single_stermmer[u])
            final_result.append(single[u])


    final_ = single_stermmer
    for i in xrange(len(multi_stermmer)):
        if multi_stermmer[i].lower() not in final_:
            final_.append(multi_stermmer[i].lower())
            final_result.append(multi[i])
    # for wd in multi_stermmer:
    #     if wd.lower() not in single_stermmer:
    #         final_.append(wd.lower())
    return final_result

def extractKeywordsBasedOnRule(sent):
    # multi-words => three grams
    # pattern => NN, NNP, NNS, NNP NNP, II NNS, NN NN

    len_ = 0

    multi_tags = ['NN', 'NNP', 'NNS' ,'NNP NNP', 'JJ NNS', 'NN NN', 'NNP NNP NNP', 'JJ NN']
    single_tags = ['NNS', 'NN', 'NNP']
    multi_words = extract_chunks_regular_expression(sent, len_limit=4)
    single_words = getWordPattern(sent)
    multi_words = multi_words[multi_words['tag'].isin(multi_tags)]
    single_words = single_words[single_words['tag'].isin(single_tags)]
    multi_ = multi_words['word'].tolist()
    single_ = single_words['word'].tolist()
    final_words = processWords(list(multi_), list(single_))

    return final_words



def getRelevantWords(df, typ = 1):

    list_keywords = []
    for i in (df['id'].count()):
        content_raw = removeURI(df.iloc[i]['content raw'])
        list_keywords = extractKeywordsBasedOnRule(content_raw)
        d = {
            'id' : df.iloc[i]['id'],
             'keywords' : list_keywords
        }
        updateSolar(d)

def getRelevantWordsOne(twitter):

    id = twitter['id']
    txt = twitter['content']
    content_raw = removeURI(txt)
    list_keywords = extractKeywordsBasedOnRule(content_raw)
    training_list_keywords, labels = getKeywordsFromSolar()
    training_list_keywords.append(list_keywords)
    td_idf_corpus, libr = score_by_tfidf(training_list_keywords)
    row = libr.num_docs
    column = libr.dfs.__len__()
    denseMatrix = np.zeros((row, column))
    for i in xrange(row):
        normal_ = sum([tu[1] for tu in td_idf_corpus.corpus[i]])
        for u in td_idf_corpus.corpus[i]:
            denseMatrix[i][u[0]] = td_idf_corpus.obj.idfs[u[0]] * u[1] / normal_
    cluster_list = ['Political', 'Business', 'Technology', 'Social', 'Others']
    list_ = []
    for i in cluster_list:
        list_matrix = []

        for u in xrange(len(labels)):
            if labels[u] == i:
                list_matrix.append(denseMatrix[i])
        num_list = len(list_matrix)
        list_np_matrix = np.array(list_matrix)
        a = list_np_matrix.sum(axis=0)
        centroid = a / num_list
        list_.append(centroid)

    base = denseMatrix[-1]
    cosineDistance = []
    for i in xrange(len(list_)):
        check = list_matrix[i]
        cosineS = cosinSimilarity(check, base)
        cosineDistance.append(cosineS)
    index_ = cosineDistance.index(min(cosineDistance))
    final_label = cluster_list[index_]
    d = {
        'id': id,
        'keywords': list_keywords,
        'label' : final_label
    }
    updateSolar(d)

def getRelevantWordsKNN(twitter):

    id = twitter['id']
    txt = twitter['content']
    content_raw = removeURI(txt)
    list_keywords = extractKeywordsBasedOnRule(content_raw)
    training_list_keywords, labels = getKeywordsFromSolar()
    training_list_keywords.append(list_keywords)
    td_idf_corpus, libr = score_by_tfidf(training_list_keywords)
    row = libr.num_docs
    column = libr.dfs.__len__()
    denseMatrix = np.zeros((row, column))
    for i in xrange(row):
        normal_ = sum([tu[1] for tu in td_idf_corpus.corpus[i]])
        for u in td_idf_corpus.corpus[i]:
            denseMatrix[i][u[0]] = td_idf_corpus.obj.idfs[u[0]] * u[1] / normal_
    cluster_list = ['Political', 'Business', 'Technology', 'Social', 'Others']
    list_ = []
    for i in cluster_list:
        list_matrix = []

        for u in xrange(len(labels)):
            if labels[u] == i:
                list_matrix.append(denseMatrix[i])
        num_list = len(list_matrix)
        list_np_matrix = np.array(list_matrix)
        a = list_np_matrix.sum(axis=0)
        centroid = a / num_list
        list_.append(centroid)

    base = denseMatrix[-1]
    cosineDistance = []
    for i in xrange(len(denseMatrix) - 1):
        check = denseMatrix[i]
        cosineS = cosinSimilarity(check, base)
        cosineDistance.append([i, cosineS])
    cosineDistance_np = np.array(cosineDistance, dtype=[('index', int), ('distance', float)])
    cosineDistance_np.sort(axis= 1, order='distance')
    top_20 = cosineDistance[:20]
    vote = {i:0 for i in cluster_list}
    for v in top_20:
        vote[v['index']] += 1
    vote_pd = pd.DataFrame.from_dict(vote, column=['label', 'vote'])
    vote_pd.sort(columns=['vote'], ascending=False)
    final_label = vote_pd.iloc[0]['label']

    d = {
        'id': id,
        'keywords': list_keywords,
        'label' : final_label
    }
    updateSolar(d)



def cosinSimilarity(array1, array2):
    from scipy.spatial.distance import cosine
    return cosine(array1, array2)

def knnPredict(matrix, k,  baseIndex, vote_decision):
    #vote decision, normalized
    matrix_ = np.transpose(matrix)
    res = []
    base = matrix_[baseIndex]
    for i in xrange(len(matrix_)):
        check = matrix_[i]
        cosineS = cosinSimilarity(check, base)
        res.append([i, cosineS])
    data = pd.DataFrame.from_records(res, columns=['term index', 'cosineS'])
    data = data[data['term index'] != baseIndex]
    data1 = data.sort(columns=['cosineS'], ascending=True)

    top20Data = data.iloc[:k]
    votes = [  vote_decision[0, data1.iloc[u]['term index'] ] for u in xrange(k) ]
    vote = sum(votes)
    if vote >= 0:
        return 1
    else:
        return 0

def score_by_tfidf(boc_words):
    import gensim, nltk
    # make gensim dictionary and corpus
    dictionary = gensim.corpora.Dictionary(boc_words)
    corpus = [dictionary.doc2bow(boc_text) for boc_text in boc_words]
    # transform corpus with tf*idf model
    tfidf = gensim.models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]

    return corpus_tfidf, dictionary






def mainWork():
    from extract_keywords import score_keyphrases_by_tfidf
    mongodb_connect = Mongodb_connect()
    mongodb_connect.dbConnect()

    data = mongodb_connect.dbQueryAll('FYP', 'fb_test2')
    df = pd.DataFrame(list(data))
    training_set = df[df['isTraining'] == 1]
    testing_set = df[df['isTraining'] == 0]
    # training_words = getRelevantWords(training_set, mongodb_connect)
    # testing_words = getRelevantWords(testing_set, mongodb_connect, typ = 2)
    training_words = training_set['token'].tolist()
    testing_words = testing_set['token'].tolist()
    k_nn = 20
    final_prediction = []

    for i in xrange(len(testing_words)):
        data = {}
        ser = testing_set.iloc[i]
        ser_ = testing_words[i]
        print ser
        training_set = training_set.append(ser, ignore_index = True)
        training_words.append(ser_)
        print training_set.iloc[-1]
        # get the clustering by DBSCAN and Kmeans
   #     dbscanData, dbscanb =jaccardVectorDBSCAN(training_set)
 #       kmeans_n_cluster, kmeans_cluster = kmeansClusteringwithAllFields(training_set)
        kmeans_cluster = jaccardCluster(training_set)
        print kmeans_cluster[-1]
        cluster_ind = kmeans_cluster[-1]
        data['cluster'] = cluster_ind
        data['num of extracted keywords'] = len(ser_)
        data['testing ad id'] = testing_set.iloc[i]['index']
        data['sequence'] = i

        loc = [i for i, u in enumerate(kmeans_cluster) if u == cluster_ind]
        loc_index = [training_set.iloc[i]['index'] for i, u in enumerate(kmeans_cluster) if u == cluster_ind]
        data['no of ads in same cluster'] = len(loc)
        training_word_ = [ training_words[i] for i in loc]
        td_idf_corpus, libr = score_by_tfidf(training_word_)
        print 2
        row = libr.num_docs
        column = libr.dfs.__len__()
        denseMatrix = np.zeros((row, column))
        for i in xrange(row):
            normal_ = sum([tu[1] for tu in td_idf_corpus.corpus[i]])
            for u in td_idf_corpus.corpus[i]:
                denseMatrix[i][u[0]] = td_idf_corpus.obj.idfs[u[0]] * u[1] / normal_
        df_training_true = training_set[training_set['index'].isin(loc_index)]['true_keyword']
        true_labels = df_training_true.tolist()
        df_training_true = training_set[training_set['index'].isin(loc_index)]['all_keywords_true_false']
        all_keywords_labels = df_training_true.tolist()


        true_labels_array = np.zeros((1, column))
        for i in xrange(row - 1):
            for u in td_idf_corpus.corpus[i]:
                term = libr[u[0]]
                if term in all_keywords_labels[i]['key_list']:
                    if term in true_labels[i]:
                        true_labels_array[0, u[0]] += 1 * u[1]
                    else:
                        true_labels_array[0, u[0]] += -1 * u[1]
        predictions = []
        prediction_key = []
        for w in td_idf_corpus.corpus[row-1]:
            result = knnPredict(denseMatrix, 20, w[0], true_labels_array)
            prediction_key.append(libr[w[0]])
            predictions.append(result)
            if result == 1:
                true_labels[row-1].append(libr[w[0]])
        final_prediction.append(predictions)
        data['prediction_value'] = predictions
        data['prediction_key'] = prediction_key
        up = {
            '_id' : ser['_id'],
            'post' : data

        }
        mongodb_connect.dbUpdate('FYP', 'fb_test2', up)

    np.savetxt("predictions.csv", final_prediction, delimiter=",")

def viewData():
    mongodb_connect = Mongodb_connect()
    mongodb_connect.dbConnect()
    a = mongodb_connect.dbQuery('FYP', 'fb_test2', {'isTraining': 0})
    df = pd.DataFrame(list(a))
    df['number of retrieved true keywords'] = 0
    df['number of retrieved false keywords'] = 0
    df['number of true keywords not retrived'] = 0
    df['number of false keywords not retrived'] = 0
    df['number of keyword candidate'] = 0


    row = df['index'].count()
    for i in xrange(row):
        ser = df.iloc[i]
        predic_value = ser['prediction_value']
        predit_key = ser['prediction_key']
        true_keyword = ser['true_keyword']
        validate_prediction = []
        for u in predit_key:

            if u.lower() in true_keyword:
                validate_prediction.append(1)
            else:
                validate_prediction.append(0)
        up = {
            '_id': ser['_id'],
            'post': {
                'actual_value' :  validate_prediction
            }

        }
        mongodb_connect.dbUpdate('FYP', 'fb_test2', up)
        no_retrived_relevant = 0
        no_retrived_false =0
        no_non_retrieved_relevant = 0
        no_non_retrieved_non_relevant = 0

        for x in xrange(len(validate_prediction)):
            if validate_prediction[x] == 1 and predic_value[x] == 1:
                no_retrived_relevant += 1
            if validate_prediction[x] == 0 and predic_value[x] == 1:
                no_retrived_false += 1
            if validate_prediction[x] == 0 and predic_value[x] == 0:
                no_non_retrieved_non_relevant += 1
            if validate_prediction[x] == 1 and predic_value[x] == 0:
                no_non_retrieved_relevant += 1

        df.set_value(i, 'number of retrieved true keywords', no_retrived_relevant)
        df.set_value(i, 'number of retrieved false keywords', no_retrived_false)
        df.set_value(i, 'number of true keywords not retrived', no_non_retrieved_relevant)
        df.set_value(i, 'number of false keywords not retrived', no_non_retrieved_non_relevant)
        df.set_value(i, 'number of keyword candidate', len(ser['token']))

    df['cummulative number of retrieved true keywords'] = df['number of retrieved true keywords'].cumsum()
    df['cummulative number of retrieved false keywords'] = df['number of retrieved false keywords'].cumsum()
    df['cummulative number of false keywords not retrived'] = df['number of false keywords not retrived'].cumsum()
    df['cummulative number of true keywords not retrived'] = df['number of true keywords not retrived'].cumsum()
    a = df[['cummulative number of retrieved true keywords','cummulative number of retrieved false keywords', 'cummulative number of false keywords not retrived',
                                                                                                              'cummulative number of true keywords not retrived']]
    a.plot()
    print a.iloc[-1]

    df['cummulative precision'] = df['cummulative number of retrieved true keywords'] / (df['cummulative number of retrieved true keywords'] + df['cummulative number of retrieved false keywords'] )
    df['cummulative recall'] = df['cummulative number of retrieved true keywords'] / (df['cummulative number of retrieved true keywords'] + df['cummulative number of true keywords not retrived'])
    df['F-measure'] = 2 * df['cummulative precision'] * df['cummulative recall'] / (df['cummulative recall'] + df['cummulative precision'])

    df.plot(x ='sequence', y = ['cummulative precision', 'cummulative recall'])
    df.plot(x ='sequence', y = 'F-measure')

    plt.show()



if __name__ == "__main__":
    viewData()






