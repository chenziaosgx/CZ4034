from django.shortcuts import render
import tweepy
import json
import solr
from urllib2 import *
from datetime import datetime


def home(request,source="",category="",sort="",page=""):

    #connect to Solr
    connection = solr.SolrConnection('http://localhost:8983/solr/CZ4034',debug=True)

    #connect to Twitter
    auth = tweepy.OAuthHandler('f11IQFNPOQuaopvynNXCjGrF3','ZwTx5EbMhjP4uDTktVINZF5E25bFjKEKWXNihiQPbzmjDg9j3S')
    auth.set_access_token('827805611865747456-WO1lplOfxP4NPMZUTOczxN3IyVz7vDc','ebekAEeMEsKRCtN5WvgMN9r4eoURI39Q0PumL2zyXelsA')
    api = tweepy.API(auth)
    #CrawlData(api,connection)

    connection = solr.SolrConnection('http://localhost:8983/solr/CZ4034',debug=True)


    if not source:
        source = "All2"
    if not category:
        category="All"

    if source != "All2":
        status_list = connection.query('name:'+source).results
    else:
        status_list = connection.query('*:*',rows = 100).results

    for status in status_list:
        status["retweet_count"] = status["retweet_count"][0]

    if sort=="Popularity":
        status_list = sorted(status_list,key=lambda status_list: status_list["like"],reverse = True)
    elif sort=="Retweet":
        status_list = sorted(status_list,key=lambda status_list: status_list["retweet_count"],reverse = True)
    else:
        status_list = sorted(status_list,key=lambda status_list: status_list["time"],reverse = True)

    request.session.status_list = status_list

    pages = getPage(request)
    if not page:
        page = 1
    request.session.status_list = getStatusList(status_list,page)

    return render(request,'home2.html',{'status_list':request.session.status_list,'source':source,'category':category,'pages':pages,'sort':sort})

def CrawlData(api,connection):
    source_list = {"Straits Times":"37874853","BBC":"742143","Wall Streets Journal":"3108351","CNN":"759251","New York Times":"807095"}
    connection.delete_query("*:*")
    for source in source_list:
        status_list = api.user_timeline(id=source_list[source],count =1000 )
        PostToSolr(status_list,connection)

def PostToSolr(status_list,connection):
    for status in status_list:
        status_json = status._json

        #Get created date
        date_list = status_json["created_at"].split(" ")
        year = date_list[-1]
        time = date_list[3]
        day = date_list[2]
        month = datetime.strptime(date_list[1],"%b").month
        profile_image = status_json["user"]["profile_image_url_https"]
        if "retweeted_status" in status_json  and "media" in status_json["retweeted_status"]["entities"]:
            tweet_image = status_json["retweeted_status"]["entities"]["media"][0]["media_url_https"]
        else:
            tweet_image = "no image"
        retweet_count = status_json["retweet_count"]
        if month<10:
            month = "0"+str(month)

        connection.add(content_raw = status_json["text"],tweet_image = tweet_image,retweet_count = retweet_count,profile_image=profile_image,id=status_json["id_str"],time=str(year)+"-"+str(month)+"-"+str(day)+"T"+time+"Z",like = status_json["favorite_count"],content=status_json["text"],name=status_json["user"]["screen_name"])



    connection.commit()

def search(request,category,source,search_value,sort="",page=""):

    search_value = search_value.replace("%20"," ")
    suggestion_list = ""
    connection = solr.SolrConnection('http://localhost:8983/solr/CZ4034',debug=True)
    if source == "All2":
        status_list = connection.query('content:"'+search_value +'"',rows=100).results
    else:
        status_list = connection.query('content:"'+search_value+'"'+' AND name:'+source).results

    conn = urlopen('http://localhost:8983/solr/CZ4034/suggest?q='+search_value.replace(" ","%20")+'&wt=json')
    suggestion_json = json.load(conn)["spellcheck"]["suggestions"]

    if len(suggestion_json)>1:
        suggestion_list = suggestion_json[1]["suggestion"]
        original_word = suggestion_json[0]

    temp_suggestion_list = []
    for suggestion in suggestion_list:
        temp_suggestion_list.append(search_value.replace(original_word,suggestion))

    suggestion_list=temp_suggestion_list

    for status in status_list:
        status["retweet_count"] = status["retweet_count"][0]

    if sort=="Popularity":
        status_list = sorted(status_list,key=lambda status_list: status_list["like"],reverse = True)
    elif sort=="Retweet":
        status_list = sorted(status_list,key=lambda status_list: status_list["retweet_count"],reverse = True)
    else:
        status_list = sorted(status_list,key=lambda status_list: status_list["time"],reverse = True)

    request.session.status_list = status_list
    pages = getPage(request)
    request.session.status_list = getStatusList(status_list,page)
    return render(request,'home2.html',{'sort':sort,'status_list':request.session.status_list,'search_value':search_value,'source':source,'category':category,'suggestion_list':suggestion_list,'pages':pages})

def getPage(request):
    length = len(request.session.status_list)/10+1
    print length
    if length>10:
        length = 10
    length = range(1,length+1)
    return length

def getStatusList(status_list,page):
    page = int(page)
    length = len(status_list)
    if (length < page*10):
        return status_list[(page-1)*10:length]
    else:
        return status_list[(page-1)*10:page*10]
# Create your views here.
