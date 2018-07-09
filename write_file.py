import tabview
import pandas
infile = "../datasets/customer-churn-model/Customer Churn Model.txt"
outfile = "../datasets/customer-churn-model/Tab Customer Churn Model.txt"

with open(infile, "r") as infile1:
    with open(outfile, "w") as outfile1:
        for line in infile1:
            fields = line.strip().split(",")
            outfile1.write("\t".join(fields))
            outfile1.write("\n")
# data = pandas.read_csv(outfile, sep="\t")
# print(data.head())
# tabview.view(outfile, start_pos=(0,40))
