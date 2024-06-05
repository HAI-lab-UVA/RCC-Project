/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 */

package com.runtasrm;

import org.apache.commons.cli.*;

import java.io.IOException;
import java.util.ArrayList;

/**
 *
 * @author andrew
 */
public class Main {

    static Options makeOptions() {
        Options options = new Options();

        Option s = new Option("s", "support", true, "minimum support");
        s.setRequired(true);
        options.addOption(s);

        Option c = new Option("c", "confidence", true, "minimum confidence");
        c.setRequired(true);
        options.addOption(c);

        Option i = new Option("i", "input", true, "path to input .txt file");
        i.setRequired(true);
        options.addOption(i);

        Option o = new Option("o", "output", true, "path to output .txt file");
        o.setRequired(true);
        options.addOption(o);

        options.addOption(Option.builder("x")
                                .longOpt("xquery")
                                .desc("antecedents for query as integer tokens")
                                .hasArgs()
                                .build());
        options.addOption(Option.builder("y")
                                .longOpt("yquery")
                                .desc("consequents for query as integer tokens")
                                .hasArgs()
                                .build());

        return options;
    }

    public static void main(String[] args) throws ParseException {
        Options options = makeOptions();
        CommandLineParser parser = new DefaultParser();
        CommandLine cmd = parser.parse(options, args);

        double minSupport = 0.0;
        double minConfidence = 0.0;
        String inputFP = "";
        String outputFP = "";

        minSupport = cmd.getParsedOptionValue("s");
        minConfidence = cmd.getParsedOptionValue("c");
        inputFP = cmd.getParsedOptionValue("i");
        outputFP = cmd.getParsedOptionValue("o");

        ArrayList<Integer> xQuery = new ArrayList<>();
        if (cmd.hasOption("x")) {
            String[] xVals = cmd.getOptionValues("x");
            for (String value : xVals) {
                xQuery.add(Integer.parseInt(value));
            }
        }

        ArrayList<Integer> yQuery = new ArrayList<>();
        if (cmd.hasOption("y")) {
            String[] yVals = cmd.getOptionValues("y");
            for (String value : yVals) {
                yQuery.add(Integer.parseInt(value));
            }
        }

        TaSRM_V3 tasrm = new TaSRM_V3();
        try {
            tasrm.runAlgorithm(minSupport, minConfidence, inputFP, outputFP, xQuery, yQuery);
        } catch (IOException e) {
            // Auto-generated catch block
            e.printStackTrace();
        }
    }
}
