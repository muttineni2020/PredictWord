# Predict word
#
# This model built by using N-Gram to predicts words
#
#
# Some useful keyboard shortcuts for package authoring:
#
#   Build and Reload Package:  'Ctrl + Shift + B'
#   Check Package:             'Ctrl + Shift + E'
#   Test Package:              'Ctrl + Shift + T'

#options( java.parameters = "-Xmx6g")


library(stringi)
library(tm)
library(RWeka)
#library(ggplot2)
library(tidyr)


print("Building the Predict model... Please wait")

url<-"https://d396qusza40orc.cloudfront.net/dsscapstone/dataset/Coursera-SwiftKey.zip"

if (!file.exists("Coursera-SwiftKey.zip")) {
  download.file(url,basename(url))
  unzip("Coursera-SwiftKey.zip")
}

# Read the blogs,news and Twitter data.
print("Reading the data... Please wait")

bData <- readLines("final/en_US/en_US.blogs.txt", encoding = "UTF-8", skipNul = TRUE)
#nData <- readLines("final/en_US/en_US.news.txt", encoding = "UTF-8", skipNul = TRUE)
tData <- readLines("final/en_US/en_US.twitter.txt", encoding = "UTF-8", skipNul = TRUE)

con <- file("final/en_US/en_US.news.txt", open="rb")
nData <- readLines(con, encoding="UTF-8")
close(con)
rm(con)

# Cleaning The Data

print("Creating sample.. Please wait")

set.seed(679)
data.sample <- c(sample(bData, length(bData) * 0.1),
                 sample(nData, length(nData) * 0.1),
                 sample(tData, length(tData) * 0.1))

# Remove all non english characters as they cause issues

print("Cleaning the data.. Please wait")

data.sample <- iconv(data.sample, "latin1", "ASCII", sub="")


# Create corpus(sample data set) and clean the data

cSample<-VCorpus(VectorSource(list(data.sample)))
cSample <- tm_map(cSample, content_transformer(tolower))
toSpace <- content_transformer(function(x, pattern) gsub(pattern, " ", x))
cSample <- tm_map(cSample, toSpace, "(f|ht)tp(s?)://(.*)[.][a-z]+")
cSample <- tm_map(cSample, toSpace, "@[^\\s]+")
cSample <- tm_map(cSample, removeWords, stopwords("english"))
cSample <- tm_map(cSample, removePunctuation)
cSample <- tm_map(cSample, removeNumbers)
cSample <- tm_map(cSample, stripWhitespace)
cSample <- tm_map(cSample, PlainTextDocument)


getFreq <- function(tm) {
  freq <- sort(rowSums(as.matrix(tm)), decreasing = TRUE)
  return(data.frame(word = names(freq), freq = freq))
}

print("Building N-Gram model.. Please wait")

uniTokenSize <- function(x) NGramTokenizer(x, Weka_control(min = 1, max = 1))
biTokenSize <- function(x) NGramTokenizer(x, Weka_control(min = 2, max = 2))
triTokenSize <- function(x) NGramTokenizer(x, Weka_control(min = 3, max = 3))
quadTokenSize <- function(x) NGramTokenizer(x, Weka_control(min = 4, max = 4))
#quintTokenSize <- function(x) NGramTokenizer(x,Weka_control(min = 5, max = 5))


print("Saving Model.. Please wait")

unigramFreq <- getFreq(removeSparseTerms(TermDocumentMatrix(cSample, control = list(tokenize = uniTokenSize)), 0.9999))
bigramFreq <- getFreq(removeSparseTerms(TermDocumentMatrix(cSample, control = list(tokenize = biTokenSize)), 0.9999))
trigramFreq <- getFreq(removeSparseTerms(TermDocumentMatrix(cSample, control = list(tokenize = triTokenSize)), 0.9999))
quadgramFreq <- getFreq(removeSparseTerms(TermDocumentMatrix(cSample, control = list(tokenize = quadTokenSize)), 0.9999))
#quintgramFreq <- getFreq(removeSparseTerms(TermDocumentMatrix(cSample, control = list(tokenize = quintTokenSize)), 0.9999))

print("Framing the data..Please Wait")

biGram <- data.frame(bigramFreq$word, separate(bigramFreq,word,c("uword","bword"),sep=" "), stringsAsFactors = FALSE)
triGram <- data.frame(trigramFreq$word, separate(trigramFreq,word,c("uword","bword","tword"),sep=" "), stringsAsFactors = FALSE)
quadGram <- data.frame(quadgramFreq$word, separate(quadgramFreq,word,c("uword","bword","tword","qword"),sep=" "), stringsAsFactors = FALSE)

colnames(biGram)[1] <- "word"
colnames(triGram)[1] <- "word"
colnames(quadGram)[1] <- "word"

print("Success fully installed - Predict Word package")
