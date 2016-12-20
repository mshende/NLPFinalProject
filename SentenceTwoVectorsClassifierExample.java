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

public class SentenceTwoVectorsClassifierExample {

    ParagraphVectors paragraphVectors;
    LabelAwareIterator iterator;
    TokenizerFactory tokenizerFactory;

    private static final Logger log = LoggerFactory.getLogger(SentenceTwoVectorsClassifierExample.class);

    public static void main(String[] args) throws Exception {

        SentenceTwoVectorsClassifierExample app = new SentenceTwoVectorsClassifierExample();
        app.makeParagraphVectors();
        app.checkUnlabeledData();
    }

    void makeParagraphVectors() throws Exception {
        ClassPathResource resource = new ClassPathResource("twovec/labeled");

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
            "he'll",
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
            "she'd",
            "she'll",
            "she's",
            "so",
            "the",
            "to",
            "too",
            "up",
            "was",
            "we",
            "we'd",
            "we'll",
            "we're",
            "we've",
            "who",
            "who's",
            "whom",
            "why",
            "why's",
            "won't",
            "you",
            "you'd",
            "you'll",
            "you're",
            "you've",
            "your",
            "yours",
            "yourself",
            "yourselves"
        ));
        // ParagraphVectors training configuration
        paragraphVectors = new ParagraphVectors.Builder()
            .learningRate(0.025)
            .minLearningRate(0.001)
            .stopWords(stopWordList)
            .batchSize(1000)
            .epochs(25)
            .iterate(iterator)
            .trainWordVectors(true)
            .useUnknown(true)
            .windowSize(6)
            .tokenizerFactory(tokenizerFactory)
            .build();

        // Start model training
        paragraphVectors.fit();
    }

    void checkUnlabeledData() throws FileNotFoundException {

        int totalNegative = 0;
        int totalNegativeCorrect = 0;
        int totalNeutralCorrect = 0;
        int totalPositive = 0;
        int totalPositiveCorrect = 0;

        ClassPathResource unClassifiedResource = new ClassPathResource("twovec/unlabeled");
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
                    case "positive":
                        totalPositive++;
                        break;
                    default:
                        log.warn("Shouldn't be in here.");
                        break;
                }

                log.info("Document '" + document.getLabel() + "' falls into the following categories: ");
                String maxScoreLabel = "no max score label";
                Double maxScore = -99999999.99999;

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

        System.out.println("Total Negative: " + totalNegative);
        System.out.println("Total Negative Correct: " + totalNegativeCorrect);
        System.out.println("Total Positive: " + totalPositive);
        System.out.println("Total Positive Correct: " + totalPositiveCorrect);

        System.out.println("\n");

        int negTruePositives = totalNegativeCorrect;
        int negFalsePositives = (totalPositive - totalPositiveCorrect);
        int negFalseNegatives = (totalNegative - totalNegativeCorrect);
        double negPrecision = (double)negTruePositives / (negTruePositives + negFalsePositives);
        double negRecall = (double)negTruePositives / (negTruePositives + negFalseNegatives);
        double alpha = 0.5;
        double negFMeasure = 1.0 / ( (alpha * (1 / negPrecision) ) + ( (1 - alpha) * ( 1 / negRecall) ) );
        System.out.println("Negative Precision");
        System.out.println("true positives: " + negTruePositives);
        System.out.println("false positives: " + negFalsePositives);
        System.out.println("negative precision = " + negPrecision);
        System.out.println("Negative Recall");
        System.out.println("true positives: " + negTruePositives);
        System.out.println("false positives: " + negFalsePositives);
        System.out.println("negative recall = " + negRecall);
        System.out.println("Negative F-measure");
        System.out.println("negative f-measure = " + negFMeasure);

        System.out.println("\n");

        int posTruePositives = totalPositiveCorrect;
        int posFalsePositives = (totalNegative - totalNegativeCorrect);
        int posFalseNegatives = (totalPositive - totalPositiveCorrect);
        double posPrecision = (double)posTruePositives / (posTruePositives + posFalsePositives);
        double posRecall = (double)posTruePositives / (posTruePositives + posFalseNegatives);
        double posFMeasure = 1.0 / ( (alpha * (1 / posPrecision) ) + ( (1 - alpha) * ( 1 / posRecall) ) );
        System.out.println("Positive Precision");
        System.out.println("true positives: " + posTruePositives);
        System.out.println("false positives: " + posFalsePositives);
        System.out.println("positive precision = " + posPrecision);
        System.out.println("Positive Recall");
        System.out.println("true positives: " + posTruePositives);
        System.out.println("false positives: " + posFalsePositives);
        System.out.println("positive recall = " + posRecall);
        System.out.println("Positive F-measure");
        System.out.println("positive f-measure = " + posFMeasure);

        System.out.println("\n");

        System.out.println("Total Accuracy = " +
            (double) (totalNegativeCorrect + totalPositiveCorrect) /
                (totalNegative + totalPositive));
        System.out.println("Negative Sentiment Accuracy = "
            + (double) totalNegativeCorrect / totalNegative);
        System.out.println("Positive Sentiment Accuracy = "
            + (double) totalPositiveCorrect / totalPositive);
        System.out.println("\n");
        System.out.println("Total Error Rate = "
            + (double) ((totalNegative + totalPositive) -
            (totalNegativeCorrect + totalNeutralCorrect + totalPositiveCorrect))
            / (totalNegative + totalPositive));
        System.out.println("Negative Sentiment Error Rate = "
            + (double) (totalNegative - totalNegativeCorrect) / totalNegative);
        System.out.println("Positive Sentiment Error Rate = "
            + (double) (totalPositive - totalPositiveCorrect) / totalPositive);

    }
}
