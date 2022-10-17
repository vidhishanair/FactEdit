#!/usr/bin/env python

import sys
import itertools
import math
import spacy

inf = float('inf')
SUB='SUB'
INS='INS'
DEL='DEL'
MATCH='-'
# spacy_nlp = spacy.load('en_core_web_lg')

class Distance(object):
    def SubstitutionCost(self, ref_element, hyp_element):
        return 1.0

    def DeletionCost(self, ref_element):
        return 1.0

    def InsertionCost(self, hyp_element):
        return 1.0

    def BuildCostMatrix(self, ref, hyp):
        d= [[(inf, MATCH) for j in range(1+len(hyp))] for i in range(1+len(ref))]
        d[0][0]=(0, MATCH) # each cell in matrix contains a cumulative path weight and a backpointer
        for i in range(1, len(ref)+1):
            d[i][0] = (d[i-1][0][0] + 1, DEL)
        for j in range(1, len(hyp)+1):
            d[0][j] = (d[0][j-1][0] + 1, INS)
        for i in range(1, len(ref)+1):
            for j in range(1, len(hyp)+1):
                if hyp[j-1] == ref[i-1]:
                    match_sub = (d[i-1][j-1][0], MATCH)
                else:
                    sub_cost = self.SubstitutionCost(ref[i-1], hyp[j-1])
                    match_sub = (d[i-1][j-1][0] + sub_cost, SUB)
                del_cost = self.DeletionCost(ref[i-1])
                ins_cost = self.InsertionCost(hyp[j-1])
                candidates = [match_sub,
                              (d[i-1][j][0] + del_cost, DEL),
                              (d[i][j-1][0] + ins_cost, INS)]
                d[i][j] = min(candidates)
        optimal_cost = d[len(ref)][len(hyp)][0]
        return d, optimal_cost

    def Visualize(self, ref, hyp, out_file) :
        # find best alignment using Levenshtein algorithm
        refAligned, hypAligned, errList, optimalCost =  self.AlignStrings(ref, hyp)
        # print output
        out_file.write( 'REF:'+' '.join(['%-8s' % x for x in refAligned])+'\n')
        out_file.write( 'HYP:'+' '.join(['%-8s' % x for x in hypAligned])+'\n')
        out_file.write( '    '+' '.join(['%-8s' % x for x in errList])+'\n')
        out_file.write( 'WER = '+str(100.0*optimalCost/len(ref))+ '%'+'\n\n')
        refop = ' '.join(['%-8s' % x for x in refAligned])
        hypop = ' '.join(['%-8s' % x for x in hypAligned])
        errop = ' '.join(['%-8s' % x for x in errList])
        wer = str(100.0*optimalCost/len(ref))
        return refop, hypop, errop, wer

    def AlignStrings(self, ref, hyp):
        d, optimal_cost = self.BuildCostMatrix(ref, hyp)
        i, j = len(ref), len(hyp)

        # get the aligned strings
        hypAligned = []
        refAligned = []
        errList = []
        while (i>0 or j>0):
            assert i>=0
            assert j>=0
            if d[i][j][1]==DEL:
                i-=1
                hypAligned.insert(0, '')
                refAligned.insert(0, ref[i])
                errList.insert(0, DEL)
            elif d[i][j][1]==INS:
                j-=1
                hypAligned.insert(0, hyp[j])
                refAligned.insert(0, '')
                errList.insert(0, INS)
            elif d[i][j][1]==SUB:
                i-=1
                j-=1
                hypAligned.insert(0, hyp[j])
                refAligned.insert(0, ref[i])
                errList.insert(0, SUB)
            else:
                i-=1
                j-=1
                hypAligned.insert(0, hyp[j])
                refAligned.insert(0, ref[i])
                errList.insert(0, MATCH)
        return refAligned, hypAligned, errList, optimal_cost

def main(args):
    #_, ref, hyp = args
    ref = "The flight data recorder, or black box, was found Thursday by a female police officer."# The French air accident investigation agency says its teams have begun to investigate its contents. A German official says Andreas Lubitz's actions amount to premeditated murder German prosecutors say a tablet found at his apartment shows he researched suicide methods."
    hyp = "The flight data recorder, or black box, was found Thursday by recovery teams."# The French air accident investigation agency says its teams have begun to investigate its contents. A German official says Andreas Lubitz's actions amount to premeditated murder German prosecutors say a tablet found at his apartment shows he researched suicide methods."
    # ref1 = ["The", "flight", "man", "was"]
    # hyp1 = ["The", "fight", "is"]
    Distance().Visualize(ref.split(), hyp.split(), sys.stdout)
    # doc = spacy_nlp(ref)
    # ref_sents = []
    # for sent in doc.sents:
    #     text = sent.text
    #     ref_sents.append(text)
    # doc = spacy_nlp(hyp)
    # hyp_sents = []
    # for sent in doc.sents:
    #     text = sent.text
    #     hyp_sents.append(text)
    # for rs, hs in zip(ref_sents, hyp_sents):
    #     Distance().Visualize(rs, hs, sys.stdout)

if __name__ == '__main__':
    main(sys.argv)