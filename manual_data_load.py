def dataCleaning():
    import pandas
    import requests

    medals_url = "http://winterolympicsmedals.com/medals.csv"

    # Basic GET request to obtain the data. The data is obtained in bytes
    # we encode it in UTF-8 to turn it into a basic string
    str_data = requests.get(medals_url).content.decode("utf-8")

    # The string is turned into a list of strings divided by each enter-jump char
    raw_data = str_data.split("\n")

    # We obtain the columns by dividing the first element of the list since we know
    # those are the headers
    columns = raw_data[0].split(",")
    len_columns = len(columns)

    # The rows begin from index 1 since index 0 represents the headers of the dataset
    rows_data = raw_data[1:]
    len_rows = len(rows_data)


    # We create a dict with the headers as keys since Pandas requires this for The
    # data frame
    main_dict = {}
    for column in columns:
        main_dict[column] = []

    # Each element in the list of rows is divided by commas and then added to the
    # respective column key in the main dictionary
    for row in rows_data:
        row_values = row.split(",")
        for index in range(len_columns):
            main_dict[columns[index]].append(row_values[index])

    print("The file has %d rows and %d columns"%(len_rows, len_columns))

    # We turn the dictionary into a Pandas.DataFrame and then save it as a .csv file
    medals_df = pandas.DataFrame(main_dict)
    medals_df.to_csv("../datasets/athletes/manual_load.csv")
    # medals_df.to_json("../datasets/athletes/manual_load.json")
    # medals_df.to_xls("../datasets/athletes/manual_load.xls")
    return
