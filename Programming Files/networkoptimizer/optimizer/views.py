from django.shortcuts import render
import pandas as pd
import numpy as np

# Create your views here.

# Cache the processed data
first_df = None
second_df = None
provider_info = None
contract_info = None
enrolment_info = None
provider_data = None
contract_info_col_need = None
contract_info_col_needed = None
feature_df = None
county_provider_data = None
first_result = True
second_result = True
first_cost = 0
first_avg_score = 0
second_cost = 0
enrollment = 0


def home(request):
    # read file - takes a few seconds. Patience is appreciated
    global provider_info
    global contract_info
    global enrolment_info
    global provider_data
    global contract_info_col_needed
    global enrolment_info_col_needed
    global first_result
    global second_result
    provider_info = pd.read_excel('optimizer/static/dataset/Marked ProviderInfo_Download_HCS Comments.xlsx')
    contract_info = pd.read_csv('optimizer/static/dataset/Marked_CPSC_Contract_Info_2018_12.csv', encoding='ISO-8859-1')
    enrolment_info = pd.read_csv('optimizer/static/dataset/Marked_CPSC_Enrollment_Info_2018_12.csv')

    # assumes first sheet in the excel workbook is contains the data of interest(We can select sheet name now.)
    # assumes the first row in excel sheet contains the column headers(It is now.)
    # read file - takes a few seconds. Patience is appreciated
    xls_puf = pd.ExcelFile('optimizer/static/dataset/Marked_SNF PUF - Provider Final 2016_HCS Clear Header.xlsx')
    snf_puf = pd.read_excel(xls_puf, "Provider")

    # drop un-needed columns from all datasets

    # the columns to be kept in provider_info
    provider_info_columns = ['PROVNUM', 'PROVNAME', 'BEDCERT', 'RESTOT', 'INHOSP', 'OVERALL_RATING', 'STATE',
                             'COUNTY_NAME']
    # keep the columns needed in provider_info
    provider_info_col_needed = provider_info[provider_info_columns]

    # the columns to be kept in snf_puf file
    snf_puf_columns = ['Provider ID', 'Total SNF Medicare Payment Amount', 'Average Length of Stay (Days)',
                       'Average HCC Score']
    # keep the columns needed in snf_puf
    snf_puf_col_needed = snf_puf[snf_puf_columns]

    # the columns to be kept in contract info
    contract_info_columns = ['Contract ID', 'Organization Marketing Name']
    # keep the columns needed in contract info
    contract_info_col_needed = contract_info[contract_info_columns]
    # we are only interested in Highmark
    contract_info_col_needed = contract_info_col_needed[
        contract_info_col_needed['Organization Marketing Name'].str.lower()
        == 'highmark senior health company'].drop_duplicates(subset=['Contract ID', 'Organization Marketing Name'])

    # the columns to be kept in enrolment info
    enrolment_info_columns = ['Contract Number', 'Plan ID', 'State', 'County', 'Enrollment']
    # keep the columns needed in enrolment info
    enrolment_info_col_needed = enrolment_info[enrolment_info_columns]
    # For the base model, only consider Allegheny
    enrolment_info_col_needed = enrolment_info_col_needed[(enrolment_info_col_needed['State'].str.lower() == 'pa')]
    # Join provider info and snf_puf tables
    # Convert joining column types to strings
    provider_info_col_needed['PROVNUM'] = provider_info_col_needed['PROVNUM'].astype(str)
    snf_puf_col_needed['Provider ID'] = snf_puf_col_needed['Provider ID'].astype(str)
    # joining tables
    provider_data = pd.merge(provider_info_col_needed, snf_puf_col_needed,
                             left_on='PROVNUM', right_on='Provider ID')

    return render(request, 'index.html', context={"first_result": first_result, "second_result": second_result})


def optimize(request):
    county = request.GET['county']
    cm = float(request.GET['cm'])
    ui = float(request.GET['ui'])
    turnover = float(request.GET['turnover'])
    min_rating = int(request.GET['min_rating'])
    to_delete = request.GET['to_delete']
    to_add = request.GET['to_add']
    to_delete = to_delete.replace(' ', '').split(';')
    to_add = to_add.replace(' ', '').split(';')
    county = 'Allegheny'
    global contract_info_col_needed
    global enrolment_info_col_needed
    global feature_df
    global enrollment
    global county_provider_data
    global first_df
    global second_df
    global first_cost
    global second_cost
    global first_avg_score
    global first_result
    second_result = True
    second_avg_score = 0
    first = False
    if feature_df is None:
        first = True
        enrolment_info_col_needed = enrolment_info_col_needed[
            (enrolment_info_col_needed['County'] == county)]
        # Join contract_info and enrolment_info columns
        # Convert joining column types to strings
        contract_info_col_needed['Contract ID'] = contract_info_col_needed['Contract ID'].astype(str)
        enrolment_info_col_needed['Contract Number'] = enrolment_info_col_needed['Contract Number'].astype(str)
        # joining tables
        enrolment_data = pd.merge(contract_info_col_needed, enrolment_info_col_needed, left_on='Contract ID',
                                  right_on='Contract Number')
        # rename county and state columns
        enrolment_data = enrolment_data.rename(columns={'County': 'COUNTY_NAME', 'State': 'STATE'})
        # Process enrolment data
        enrolment_data['Enrollment'] = enrolment_data['Enrollment'].replace('*', 0)
        enrolment_data['Enrollment'] = enrolment_data['Enrollment'].astype('int')
        enrollment_by_county = enrolment_data.groupby(['STATE', 'COUNTY_NAME'])['Enrollment'].sum()
        enrollment_by_county = pd.DataFrame(enrollment_by_county)
        enrollment_by_county = enrollment_by_county.rename(columns={'Enrollment': 'ENROLLMENT'}).reset_index()[
            ['STATE', 'COUNTY_NAME', 'ENROLLMENT']]
        print(enrollment_by_county)
        enrollment = enrollment_by_county['ENROLLMENT'][0]

        # data = pd.merge(provider_data, enrolment_data, on=['STATE', 'COUNTY_NAME'])

        # build the feature matrix
        filename = county + '.csv'
        distance_info = pd.read_csv('optimizer/static/distance/' + filename, encoding='ISO-8859-1')

        distance_info['origin_id'] = distance_info['origin_id'].astype('str')

        county_provider_data = provider_data[
            (provider_data['STATE'] == 'PA') & (provider_data['COUNTY_NAME'] == county)]

        county_provider_data = county_provider_data.reset_index(drop=True)
        county_provider_data = county_provider_data.rename(columns={'Total SNF Medicare Payment Amount': 'COST'})
        # debug
        # print(county_provider_data)

        distance_matrix = []
        for i in range(len(county_provider_data)):
            distance_arr = []
            original_id = county_provider_data.iloc[i, 0]
            distance_arr.append(original_id)
            # print("original", original_id)
            for j in range(len(county_provider_data)):
                if i == j:
                    distance_arr.append(0.0)
                    continue
                destination_id = county_provider_data.iloc[j, 0]
                # print("destination", destination_id)
                distance_df = distance_info[((distance_info['origin_id'] == original_id) &
                                             (distance_info['destination_id'] == destination_id)) |
                                            ((distance_info['origin_id'] == destination_id) &
                                             (distance_info['destination_id'] == original_id))]
                #         print(distance_df)
                if distance_df.iloc[0, 10] > 20:
                    distance_arr.append(0.0)
                else:
                    distance_arr.append(1.0)
            # print(distance_arr)
            distance_matrix.append(distance_arr)

        distance_matrix = np.array(distance_matrix)
        # distance_matrix.shape
        distance_df = pd.DataFrame(distance_matrix)
        distance_df.rename(columns={distance_df.columns[0]: "PROVNUM"}, inplace=True)
        all_feature_df = pd.merge(county_provider_data, distance_df, on=['PROVNUM'])

        distance_index = list(range(12, 12 + len(all_feature_df)))
        index = [2, 5, 9]
        index = index + distance_index
        # print(index)
        # all_feature_df[list(range(1,60))] = all_feature_df[list(range(1,60))].astype('float')
        feature_df = all_feature_df.iloc[:, index]

        all_feature_df[list(range(1, 60))] = all_feature_df[list(range(1, 60))].astype('float')

    model_df, costs = reform_df(feature_df, county_provider_data, to_delete, to_add)

    # Build constraint matrix
    # min_rating = min_rating
    A = []
    # Bed number constraint
    A.append(model_df['BEDCERT'].values.T)
    # Overall rating constraint
    sample_size = len(model_df)
    for i in range(sample_size):
        tmp = np.zeros(sample_size)
        tmp[i] = -min_rating
        A.append(tmp)
    # Neighbor constraint
    for i in range(sample_size):
        tmp = model_df.iloc[i, 3:].values.astype('float64')
        tmp[i] = -1
        A.append(tmp)

    #  Build the upper bound (right side) of the inequality

    B = []
    # Bed number upper bound
    B.append((enrollment * cm * ui * turnover / 365))

    # Overall rating upper bound
    sample_size = len(model_df)
    for i in range(sample_size):
        tmp = -model_df['OVERALL_RATING'][i]
        B.append(tmp)
    # Neighbor upper bound
    for i in range(sample_size):
        B.append(0)

    # Build the object function
    objective_func = model_df['COST'].values.T

    import cvxpy
    selection = cvxpy.Variable((sample_size), boolean=True, integer=True)

    total_cost = np.array(objective_func) * selection

    # constraints = np.array([a * selection <= B for a in np.array(A)])
    B = np.array(B).reshape(119, 1)
    constraints = [A[i] * selection >= B[i] for i in range(60)]
    # constraints = feature_df['BEDCERT'].values.T * selection >= 5000

    network_optimization = cvxpy.Problem(cvxpy.Minimize(total_cost), constraints)

    network_optimization.solve(solver=cvxpy.GLPK_MI)

    # print(total_cost.value + costs)
    print('enrollment', enrollment)
    print('capacity constraints', enrollment * cm * ui * turnover / 365)

    selected_providers = county_provider_data[(selection.value == 1) | county_provider_data['PROVNUM'].isin(to_add)]

    # cache to global variable
    if first:
        first_df = selected_providers
        first_avg_score = first_df['OVERALL_RATING'].mean()
        if total_cost.value is None:
            first_result = False
            feature_df = None
        else:
            first_cost = total_cost.value + costs
            feature_df = model_df
    else:
        second_df = selected_providers
        second_avg_score = second_df['OVERALL_RATING'].mean()
        if total_cost.value is None:
            second_result = False
        else:
            second_cost = total_cost.value + costs
    print("first score", first_avg_score)
    print("second score", second_avg_score)
    print("first cost", first_cost)
    print("second cost", second_cost)
    print("first result", first_result)
    print("second result", second_result)
    return render(request, 'index.html',
                  context={"first_df": first_df, "second_df": second_df, "first_avg_score": first_avg_score,
                           "second_avg_score": second_avg_score, "first_cost": first_cost, "second_cost": second_cost, "first_result": first_result, "second_result": second_result})


def reform_df(df, county_data, to_delete, to_add):
    costs = 0
    final_df = df.copy()
    #     Deal with to_delete first
    if to_delete:
        deleted_index = county_data[county_data['PROVNUM'].isin(to_delete)].index
        for i in deleted_index:
            final_df.loc[i, 'OVERALL_RATING'] = 0

    if to_add:
        added_index = county_data[county_data['PROVNUM'].isin(to_add)].index
        for i in added_index:
            final_df.loc[i, 'COST'] = 0

        for i in added_index:
            costs += df['COST'][i]
            print("cost", costs)

    return final_df, costs
