![Uploading image.png…]()

![Uploading image.png…]()


explanation:
the model is trained on a dataset which has a column of messages and another column which says whether each message is spam or ham(not spam)
the data is then processed and converted into a machine readable language.
words are stemmed down, which means a word like PLAYING will turn into PLAY and LOVING will turn into LOVE

What is TF-IDF? Term Frequency - Inverse Document Frequency (TF-IDF) is a widely used statistical method in natural language processing and information retrieval. It measures how important a term is within a document relative to a collection of documents (i.e., relative to a corpus).
TF: term frequency, how often a word shows up
IDF: inverse document frequency, how special or unique a word is. If a word shows up in a lot of messages (like A and THE) then it's not special, but if a word is rare and doesn't appear in many messages (like CONGRATULATIONS) then that means it's important.
TF-IDF: We combine these two things to figure out which words are really important in a message. If a word shows up a lot in one message but isn't common in other messages, it gets a high score. If it's common in many messages, it gets a lower score.
So, TF-IDF helps us figure out which words are special and important in each message, and that helps us decide if a message is spam or not.
so, the model learns to pay attention to words that show up a lot in one message but are rare overall. This helps it figure out which messages are likely spam and which ones aren't. It's like teaching the model to spot patterns in the words people use to trick us with spam messages.

after figuring out which words are important in each message using TF-IDF, the model needs a way to make a final decision about whether a message is spam or not. That's where Multinomial Naive Bayes comes in. it's a maths formula
it looks at all the important words in a message and calculates the probability that the message is spam based on how often those words appear in spam messages compared to non-spam messages. 
then, it compares this probability with the probability of the message being non-spam.
by doing this for each message, the model learns to make a guess about whether a new message is spam or not by comparing these probabilities. It's called 'naive' because it makes a simple assumption that the words in the message are independent of each other, even though that might not always be true.
