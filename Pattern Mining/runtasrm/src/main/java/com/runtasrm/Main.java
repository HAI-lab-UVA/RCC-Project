package com.runtasrm;

import org.apache.commons.cli.*;

import java.io.IOException;
import java.util.ArrayList;

/**
 *
 * @author Andrew Balch
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

        options.addOption(new Option("na", "numante", true, "max number of antecedents as an integer (> 0, >= length of xquery)"));
        options.addOption(new Option("nc", "numcons", true, "max number of consequents as an integer (> 0, >= length of yquery)"));

        options.addOption(Option.builder("x")
                                .longOpt("xquery")
                                .desc("antecedents for query as integer tokens")
                                .hasArgs()
                                .required(false)
                                .build());
        options.addOption(Option.builder("y")
                                .longOpt("yquery")
                                .desc("consequents for query as integer tokens")
                                .hasArgs()
                                .required(false)
                                .build());

        return options;
    }

    public static void main(String[] args) throws ParseException {
        // TODO: Support setting max ant. and cons. sizes
        Options options = makeOptions();
        CommandLineParser parser = new DefaultParser();
        CommandLine cmd = parser.parse(options, args);

        double minSupport = Double.parseDouble(cmd.getOptionValue("s"));
        double minConfidence = Double.parseDouble(cmd.getOptionValue("c"));
        String inputFP = cmd.getParsedOptionValue("i");
        String outputFP = cmd.getParsedOptionValue("o");

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

        if (cmd.hasOption("na")) {
            int numAnte = Integer.parseInt(cmd.getOptionValue("na"));
            if (numAnte >= xQuery.size() && numAnte > 0) {
                tasrm.maxAntecedentSize = numAnte;
            }
            else {
                System.out.println("Ignoring argument for numante, must be integer >= xQuery and > 0");
            }
        }
        if (cmd.hasOption("nc")) {
            int numCons = Integer.parseInt(cmd.getOptionValue("nc"));
            if (numCons >= yQuery.size() && numCons > 0) {
                tasrm.maxConsequentSize = numCons;
            }
            else {
                System.out.println("Ignoring argument for numcons, must be integer >= yQuery and > 0");
            }
        }

        try {
            tasrm.runAlgorithm(minSupport, minConfidence, inputFP, outputFP, xQuery, yQuery);
        } catch (IOException e) {
            // Auto-generated catch block
            e.printStackTrace();
        }
    }
}
