from flask import Flask, render_template, flash, request
from rkmf import RKMFAlgorithm 
import pandas as pd
from SPARQLWrapper import SPARQLWrapper, JSON
import json
from surprise import Dataset
from surprise import Reader
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField, SelectField


DEBUG = True
app = Flask(__name__)
app.config['SECRET_KEY'] = '7d671f27d441z24367d481f2b6176p'

movies_df = pd.read_csv('./data/moviesWithDBpedia.csv')
similarities = pd.read_csv('./data/movies_tf_idf_similarities.csv',names= ['movie_1','movie_2','similarity'])
reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1, rating_scale=(0.5, 5.0))
#data = Dataset.load_from_file('./data/validationset.csv', reader=reader)
#trainset = data.build_full_trainset()
rkmf = RKMFAlgorithm(n_epochs=1, n_factors=10)
print('Comienza el fit')
#rkmf.fit(trainset)
print('ya hizo el fit')
ratings = pd.read_csv('./data/validationSet.csv')

class ReusableForm(Form):
    user = TextField('Usuario:', validators=[validators.required()])
    sr = SelectField('SR', validators=[validators.required()])
@app.route("/", methods=['GET', 'POST'])
def index():
    form = ReusableForm(request.form)
    recomendacionesOntologia = []
    recomendacionesContenido = []
    ratingsPipe = []
    if request.method == 'POST':
        user = request.form['user']
        sr = request.form['sr']
        print(sr)
        if sr == 'ontologia':
            recomendacionesOntologia = getUserOntologyRecomendations(int(user))
        if sr == 'contenido':
            recomendacionesContenido = getContentRecomendation(int(user))
        if sr == 'pipeline':
            recomendacionesOntologia = getUserOntologyRecomendations(int(user))
            print(recomendacionesOntologia)
            for recomendacion in recomendacionesOntologia:
                ratingsPipe.append(getUserRKMFRating(int(user),recomendacion))
            print(ratingsPipe)
    return render_template('index.html', recomendacionesOntologia = recomendacionesOntologia, ratingsPipe = ratingsPipe, recomendacionesContenido = recomendacionesContenido)

def getMovieOntologyRecomendations(movie_id):
    link = movies_df[movies_df['movieId'] == movie_id]['dbpediaLink'].values[0]
    recomendations = []
    if str(link) != 'nan':
        sparql = SPARQLWrapper("http://dbpedia.org/sparql")
        sparql.setReturnFormat(JSON)
        prefix = "PREFIX owl: <http://www.w3.org/2002/07/owl#> PREFIX xsd: <http://www.w3.org/2001/XMLSchema#> PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> PREFIX foaf: <http://xmlns.com/foaf/0.1/> PREFIX dc: <http://purl.org/dc/elements/1.1/> PREFIX : <http://dbpedia.org/resource/> PREFIX dbpedia2: <http://dbpedia.org/property/> PREFIX dbpedia: <http://dbpedia.org/> PREFIX skos: <http://www.w3.org/2004/02/skos/core#>"
        sparql.setQuery(prefix+"SELECT  ?subject WHERE { <"+link+"> dct:subject ?subject .} ")
        query = prefix+"SELECT  ?subject WHERE {"
        data = sparql.query().convert()
        for subject in data['results']['bindings']:        
            query = query +"OPTIONAL{?subject dct:subject <" + subject['subject']['value'] + "> }"
        sparql.setQuery(query+"} LIMIT 3")
        data2 = sparql.query().convert()
        for subject2 in data2['results']['bindings']:
            if subject2['subject']['value'] != link:
                recomendations.append(subject2['subject']['value'])
    return recomendations

def getUserOntologyRecomendations(user_id):
    movies = ratings[ratings['userId'] == user_id]['movieId'].values
    ratingsMovies = ratings[ratings['userId'] == user_id]['rating'].values
    noEcontrado = True
    recomendations = []
    for idx, movie in enumerate(movies):
        if ratingsMovies[idx] >=4 and noEcontrado:
            link = movies_df[movies_df['movieId'] == movie]['dbpediaLink'].values[0]
            if str(link) != 'nan':
                sparql = SPARQLWrapper("http://dbpedia.org/sparql")
                sparql.setReturnFormat(JSON)
                prefix = "PREFIX owl: <http://www.w3.org/2002/07/owl#> PREFIX xsd: <http://www.w3.org/2001/XMLSchema#> PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> PREFIX foaf: <http://xmlns.com/foaf/0.1/> PREFIX dc: <http://purl.org/dc/elements/1.1/> PREFIX : <http://dbpedia.org/resource/> PREFIX dbpedia2: <http://dbpedia.org/property/> PREFIX dbpedia: <http://dbpedia.org/> PREFIX skos: <http://www.w3.org/2004/02/skos/core#>"
                sparql.setQuery(prefix+"SELECT  ?subject WHERE { <"+link+"> dct:subject ?subject .} ")
                query = prefix+"SELECT  ?subject WHERE {"
                data = sparql.query().convert()
                for subject in data['results']['bindings']:        
                    query = query +"OPTIONAL{?subject dct:subject <" + subject['subject']['value'] + "> }"
                sparql.setQuery(query+"} LIMIT 3")
                data2 = sparql.query().convert()
                for subject2 in data2['results']['bindings']:
                    if any(subject2) and subject2['subject']['value'] != link:
                        recomendations.append(subject2['subject']['value'].replace('dbpedia.org/resource/','wikipedia.org/wiki/'))
                        noEcontrado = False
    return recomendations

def getUserRKMFRating(user_id,movie_link):
    movie_link = movie_link.replace('wikipedia.org/wiki/','dbpedia.org/resource/')
    if any(movies_df[movies_df['dbpediaLink'] == movie_link]['movieId']):
        movie_id = movies_df[movies_df['dbpediaLink'] == movie_link]['movieId'].values[0]
        return rkmf.predict(user_id,movie_id).est
    return 'No se encuentra en el dataset'

def getContentRecomendation(user_id):
    recomendations = []
    movies = ratings[ratings['userId'] == user_id]['movieId'].values
    for movie in movies:
        resultado = similarities[similarities['movie_1'] == movie].values
        for r in resultado:
            if r[2] > 0.9:
                recomendations.append(movies[r[1]]['title'])
    return recomendations