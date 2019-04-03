from django.shortcuts import render
import pandas as pd
import numpy as np

# Create your views here.

# Cache the processed data
df_before = None
df_after = None
provider_info = None
contract_info = None
enrolment_info = None
provider_data = None
contract_info_col_need = None
contract_info_col_needed = None


def home(request):
    # read file - takes a few seconds. Patience is appreciated
    global provider_info
    global contract_info
    global enrolment_info
    global provider_data
    global contract_info_col_needed
    global enrolment_info_col_needed
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

    return render(request, 'index.html', context={})


def optimize(request, county=None, addition=None, deletion=None):
    mode = request['mode']  # Reoptimize or optimize
    county = 'Allegheny'
    global contract_info_col_needed
    global enrolment_info_col_needed
    enrolment_info_col_needed = enrolment_info_col_needed[(enrolment_info_col_needed['County'].str.lower() == county)]
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

    # data = pd.merge(provider_data, enrolment_data, on=['STATE', 'COUNTY_NAME'])

    # build the feature matrix
    filename = county + '.csv'
    distance_info = pd.read_csv('optimizer/static/distance/' + filename, encoding='ISO-8859-1')

    distance_info['origin_id'] = distance_info['origin_id'].astype('str')

    county_provider_data = provider_data[
        (provider_data['STATE'] == 'PA') & (provider_data['COUNTY_NAME'] == county)]

    county_provider_data = county_provider_data.reset_index(drop=True)
    # debug
    print(county_provider_data)

    distance_matrix = []
    for i in range(len(county_provider_data)):
        distance_arr = []
        original_id = county_provider_data.iloc[i, 0]
        distance_arr.append(original_id)
        print("original", original_id)
        for j in range(len(county_provider_data)):
            if i == j:
                distance_arr.append(0.0)
                continue
            destination_id = county_provider_data.iloc[j, 0]
            print("destination", destination_id)
            distance_df = distance_info[((distance_info['origin_id'] == original_id) &
                                         (distance_info['destination_id'] == destination_id)) |
                                        ((distance_info['origin_id'] == destination_id) &
                                         (distance_info['destination_id'] == original_id))]
            #         print(distance_df)
            if distance_df.iloc[0, 10] > 20:
                distance_arr.append(0.0)
            else:
                distance_arr.append(1.0)
        print(distance_arr)
        distance_matrix.append(distance_arr)

    distance_matrix = np.array(distance_matrix)
    distance_matrix.shape
    distance_df = pd.DataFrame(distance_matrix)
    distance_df.rename(columns={distance_df.columns[0]: "PROVNUM"}, inplace=True)
    all_feature_df = pd.merge(county_provider_data, distance_df, on=['PROVNUM'])

    distance_index = list(range(12, 71))
    index = [2, 5, 9]
    index = index + distance_index
    # print(index)
    # all_feature_df[list(range(1,60))] = all_feature_df[list(range(1,60))].astype('float')
    feature_df = all_feature_df.iloc[:, index]

    all_feature_df[list(range(1, 60))] = all_feature_df[list(range(1, 60))].astype('float')

    # Build constraint matrix
    min_rating = 3
    A = []
    # Bed number constraint
    A.append(feature_df['BEDCERT'].values.T)
    # Overall rating constraint
    sample_size = len(feature_df)
    for i in range(sample_size):
        tmp = np.zeros(sample_size)
        tmp[i] = -min_rating
        A.append(tmp)
    # Neighbor constraint
    for i in range(sample_size):
        tmp = feature_df.iloc[i, 3:].values.astype('float64')
        tmp[i] = -1
        A.append(tmp)

    #  Build the upper bound (right side) of the inequality
    # enrollment = enrollment_by_county['ENROLLMENT'][0]
    B = []
    # Bed number upper bound
    B.append(15000 * 0.2)

    # Overall rating upper bound
    sample_size = len(feature_df)
    for i in range(sample_size):
        tmp = -feature_df['OVERALL_RATING'][i]
        B.append(tmp)
    # Neighbor upper bound
    for i in range(sample_size):
        B.append(0)

    # Build the object function
    objective_func = feature_df['Total SNF Medicare Payment Amount'].values.T

    import cvxpy
    selection = cvxpy.Variable((sample_size), boolean=True, integer=True)

    total_cost = np.array(objective_func) * selection

    # constraints = np.array([a * selection <= B for a in np.array(A)])
    B = np.array(B).reshape(119, 1)
    constraints = [A[i] * selection >= B[i] for i in range(60)]
    # constraints = feature_df['BEDCERT'].values.T * selection >= 5000

    network_optimization = cvxpy.Problem(cvxpy.Minimize(total_cost), constraints)

    network_optimization.solve(solver=cvxpy.GLPK_MI)

    print(total_cost.value)

    selected_providers = county_provider_data[selection.value == 1]

    return render(request, 'index.html', context={"df": selected_providers})


def reform_df(df, to_delete, to_add):
    to_delete = to_delete.replace(' ', '').split(';')
    to_add = to_add.replace(' ', '').split(';')
