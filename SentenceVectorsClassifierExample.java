package org.deeplearning4j.examples.nlp.paragraphvectors;

import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.examples.nlp.paragraphvectors.tools.LabelSeeker;
import org.deeplearning4j.examples.nlp.paragraphvectors.tools.MeansBuilder;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.paragraphvectors.ParagraphVectors;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.text.documentiterator.FileLabelAwareIterator;
import org.deeplearning4j.text.documentiterator.LabelAwareIterator;
import org.deeplearning4j.text.documentiterator.LabelledDocument;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.FileNotFoundException;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;


public class SentenceVectorsClassifierExample {

    ParagraphVectors paragraphVectors;
    LabelAwareIterator iterator;
    TokenizerFactory tokenizerFactory;

    private static final Logger log = LoggerFactory.getLogger(SentenceVectorsClassifierExample.class);

    public static void main(String[] args) throws Exception {

        SentenceVectorsClassifierExample app = new SentenceVectorsClassifierExample();
        app.makeParagraphVectors();
        app.checkUnlabeledData();
    }

    void makeParagraphVectors() throws Exception {
        ClassPathResource resource = new ClassPathResource("sentvec/labeled");

        // build a iterator for our dataset
        iterator = new FileLabelAwareIterator.Builder()
            .addSourceFolder(resource.getFile())
            .build();

        tokenizerFactory = new DefaultTokenizerFactory();
        tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());
        List<String> stopWordList = Collections.unmodifiableList(Arrays.asList(
            "a",
            "all",
            "am",
            "an",
            "and",
            "any",
            "are",
            "as",
            "at",
            "be",
            "but",
            "by",
            "do",
            "few",
            "for",
            "had",
            "has",
            "he",
            "he'd",
            "he's",
            "her",
            "him",
            "his",
            "how",
            "i",
            "i'd",
            "i'll",
            "i'm",
            "i've",
            "if",
            "in",
            "is",
            "it",
            "it's",
            "its",
            "me",
            "my",
            "of",
            "off",
            "on",
            "or",
            "our",
            "out",
            "own",
            "she",
            "so",
            "the",
            "to",
            "too",
            "up",
            "was",
            "we",
            "we'd",
            "who",
            "whom",
            "why",
            "you"
        ));

        // ParagraphVectors training configuration
        paragraphVectors = new ParagraphVectors.Builder()
            .learningRate(0.025)
            .minLearningRate(0.001)
            .minWordFrequency(5)
            .layerSize(500)
            .stopWords(stopWordList)
            .batchSize(1000)
            .epochs(25)
            .iterations(5)
            .iterate(iterator)
            .trainWordVectors(true)
            .useUnknown(true)
            .windowSize(8)
            .tokenizerFactory(tokenizerFactory)
            .build();

        // Start model training
        paragraphVectors.fit();
    }

    void checkUnlabeledData() throws FileNotFoundException {

        int totalNegative = 0;
        int totalNegativeCorrect = 0;
        int totalSomewhatNegative = 0;
        int totalSomewhatNegativeCorrect = 0;
        int totalNeutral = 0;
        int totalNeutralCorrect = 0;
        int totalSomewhatPositive = 0;
        int totalSomewhatPositiveCorrect = 0;
        int totalPositive = 0;
        int totalPositiveCorrect = 0;

        ClassPathResource unClassifiedResource = new ClassPathResource("sentvec/unlabeled");
        FileLabelAwareIterator unClassifiedIterator = new FileLabelAwareIterator.Builder()
            .addSourceFolder(unClassifiedResource.getFile())
            .build();

        MeansBuilder meansBuilder = new MeansBuilder(
            (InMemoryLookupTable<VocabWord>) paragraphVectors.getLookupTable(),
            tokenizerFactory);
        LabelSeeker seeker = new LabelSeeker(iterator.getLabelsSource().getLabels(),
            (InMemoryLookupTable<VocabWord>) paragraphVectors.getLookupTable());

        while (unClassifiedIterator.hasNextDocument()) {
            try {
                LabelledDocument document = unClassifiedIterator.nextDocument();
                INDArray documentAsCentroid = meansBuilder.documentAsVector(document);
                List<Pair<String, Double>> scores = seeker.getScores(documentAsCentroid);

                switch (document.getLabel()) {
                    case "negative":
                        totalNegative++;
                        break;
                    case "somewhat_negative":
                        totalSomewhatNegative++;
                        break;
                    case "neutral":
                        totalNeutral++;
                        break;
                    case "somewhat_positive":
                        totalSomewhatPositive++;
                        break;
                    case "positive":
                        totalPositive++;
                        break;
                    default:
                        log.warn("Shouldn't be in here.");
                        break;
                }

                log.info("Document '" + document.getLabel() + "' falls into the following categories: ");
                String maxScoreLabel = "no max score label";
                Double maxScore = Double.MIN_VALUE;

                for (Pair<String, Double> score : scores) {
                    log.info("        " + score.getFirst() + ": " + score.getSecond());
                    if (Double.compare(score.getSecond(), maxScore) > 0) {
                        maxScore = score.getSecond();
                        maxScoreLabel = score.getFirst();
                    }
                }

                if (document.getLabel().equals(maxScoreLabel)) {
                    log.info("CORRECT!");
                    switch (document.getLabel()) {
                        case "negative":
                            totalNegativeCorrect++;
                            break;
                        case "somewhat_negative":
                            totalSomewhatNegativeCorrect++;
                            break;
                        case "neutral":
                            totalNeutralCorrect++;
                            break;
                        case "somewhat_positive":
                            totalSomewhatPositiveCorrect++;
                            break;
                        case "positive":
                            totalPositiveCorrect++;
                            break;
                        default:
                            log.warn("Shouldn't be in here.");
                            break;
                    }
                }
            } catch (Exception e) {
                log.warn(e.getMessage());
            }

        }

        System.out.println("Total Accuracy = " +
            (double) (totalNegativeCorrect + totalSomewhatNegativeCorrect + totalNeutralCorrect + totalSomewhatPositiveCorrect + totalPositiveCorrect) /
                (totalNegative + totalSomewhatNegative + totalNeutral + totalSomewhatPositive + totalPositive));
        System.out.println("Negative Sentiment Accuracy = "
            + (double) totalNegativeCorrect / totalNegative);
        System.out.println("Somewhat Negative Sentiment Accuracy = "
            + (double) totalSomewhatNegativeCorrect / totalSomewhatNegative);
        System.out.println("Neutral Sentiment Accuracy = "
            + (double) totalNeutralCorrect / totalNeutral);
        System.out.println("Somewhat Positive Sentiment Accuracy = "
            + (double) totalSomewhatPositiveCorrect / totalSomewhatPositive);
        System.out.println("Positive Sentiment Accuracy = "
            + (double) totalPositiveCorrect / totalPositive);

        System.out.println("Total Error Rate = "
            + (double) ((totalNegative + totalSomewhatNegative + totalNeutral + totalSomewhatPositive + totalPositive) -
            (totalNegativeCorrect + totalSomewhatNegativeCorrect + totalNeutralCorrect + totalSomewhatPositiveCorrect + totalPositiveCorrect))
            / (totalNegative + totalSomewhatNegative + totalNeutral + totalSomewhatPositive + totalPositive));
        System.out.println("Negative Sentiment Error Rate = "
            + (double) (totalNegative - totalNegativeCorrect) / totalNegative);
        System.out.println("Somewhat Negative Sentiment Error Rate = "
            + (double) (totalSomewhatNegative - totalSomewhatNegativeCorrect) / totalSomewhatNegative);
        System.out.println("Neutral Sentiment Error Rate = "
            + (double) (totalNeutral - totalNeutralCorrect) / totalNeutral);
        System.out.println("Somewhat Positive Sentiment Error Rate = "
            + (double) (totalSomewhatPositive - totalSomewhatPositiveCorrect) / totalSomewhatPositive);
        System.out.println("Positive Sentiment Error Rate = "
            + (double) (totalPositive - totalPositiveCorrect) / totalPositive);
    }
}
