from django.shortcuts import render
import pandas as pd
import numpy as np

# Cache the processed data
# The current county
county_name = ''
# Cache the result from the last execution
first_df = None
first_result = True
second_result = True
first_cost = 0
first_avg_score = 0
second_cost = 0
enrollment_number = 0
first_beds = 0
second_beds = 0

# Cache the dataset read from the csv file
provider_info = None
enrollment_info = None
# Cache the intermediate dataframe
feature_df = None
county_provider_data = None
# Cache the initial input parameter
cm_init = 0
ui_init = 0
turnover_init = 0
min_rating_init = 0
feature_1_init = 0
feature_2_init = 0
feature_3_init = 0
# Cache the black/white list
blacklistStr = ''
whitelistStr = ''


def home(request):
    clear_all()
    global provider_info
    global feature_df
    global enrollment_info
    feature_df = None
    # read file - takes a few seconds. Patience is appreciated
    if provider_info is None:
        provider_info = pd.read_csv('optimizer/static/dataset/provider_data.csv')
        enrollment_info = pd.read_csv('optimizer/static/dataset/enrollment_by_county.csv')
    return render(request, 'index.html', context={"first_result": first_result, "second_result": second_result})


def reset(request):
    clear_all()
    return home(request)


def clear_all():
    global provider_info
    global enrollment_info
    global feature_df
    global county_provider_data
    global first_result
    global second_result
    global first_cost
    global second_cost
    global first_avg_score
    global first_df
    global enrollment_number
    global first_beds
    global second_beds

    provider_info = None
    enrollment_info = None
    feature_df = None
    county_provider_data = None
    first_result = True
    second_result = True
    first_df = None
    first_cost = 0
    first_avg_score = 0
    second_cost = 0
    enrollment_number = 0
    first_beds = 0
    second_beds = 0


def re_optimize(request):
    global cm_init
    global ui_init
    global turnover_init
    global min_rating_init
    global feature_1_init
    global feature_2_init
    global feature_3_init
    global county_name
    global first_beds
    global second_beds
    county = request.POST['county']
    cm = float(request.POST['cm'])
    ui = float(request.POST['ui']) / 100
    turnover = float(request.POST['turnover'])
    min_rating = int(request.POST['min_rating'])
    # Uncomment here
    # feature_1 = float(request.POST['feature_1'])
    # feature_2 = float(request.POST['feature_2'])
    # feature_3 = float(request.POST['feature_3'])
    to_delete = request.POST['to_delete']
    to_add = request.POST['to_add']
    to_delete_orig = to_delete
    to_add_orig = to_add
    to_delete = to_delete.replace(' ', '').split(';')
    to_add = to_add.replace(' ', '').split(';')

    global feature_df
    global first_df
    global provider_info
    global first_result
    global second_result
    global first_cost
    global first_avg_score
    global second_cost
    global county_provider_data
    global enrollment_number

    cm_init = request.POST['cm']
    ui_init = request.POST['ui']
    turnover_init = request.POST['turnover']
    min_rating_init = request.POST['min_rating']
    # Uncomment here
    # feature_1_init = feature_1
    # feature_2_init = feature_2
    # feature_3_init = feature_3

    if county != county_name:
        optimize(request)

    model_df, costs_1, hold_capacity_1 = change_providers(feature_df, county_provider_data, to_delete, to_add)
    black, white = read_list_from_form()
    model_df, costs_2, hold_capacity_2 = change_providers(model_df, county_provider_data, black, white)
    # Overall rating constraint
    model_df = add_constraint(model_df, min_rating, 'OVERALL_RATING', True)
    # Make changes here.
    # 'FEATURE_X' should be replaced by the column name in the dataset.
    # model_df = add_constraint(model_df, feature_1, 'FEATURE_1', True)
    # model_df = add_constraint(model_df, feature_2, 'FEATURE_2', True)
    # model_df = add_constraint(model_df, feature_3, 'FEATURE_3', True)
    model_df, selected_providers, total_cost = execute_model(cm, county, county_provider_data,
                                                             hold_capacity_1 + hold_capacity_2,
                                                             model_df, to_add, white, turnover, ui)
    second_df = selected_providers
    second_avg_score = second_df['OVERALL_RATING'].mean()
    if total_cost.value is None:
        second_result = False
    else:
        second_cost = total_cost.value + costs_1 + costs_2
    print("first score", first_avg_score)
    print("second score", second_avg_score)
    print("first cost", first_cost)
    print("second cost", second_cost)
    print("first result", first_result)
    print("second result", second_result)
    county_name = county
    return render(request, 'second_model.html',
                  context={"first_df": first_df, "second_df": second_df, "first_avg_score": round(first_avg_score, 3),
                           "second_avg_score": round(second_avg_score, 3), "first_cost": round(first_cost, 3),
                           "second_cost": round(second_cost, 3),
                           "first_result": first_result, "second_result": second_result, 'county': county,
                           'cm': cm_init,
                           'ui': ui_init, 'turnover': turnover_init, 'min_rating': min_rating_init,
                           'to_add': to_add_orig,
                           'to_delete': to_delete_orig,
                           # Uncomment here
                           # 'feature_1': feature_1_init,
                           # 'feature_2': feature_2_init,
                           # 'feature_3': feature_3_init,
                           'blacklistStr': blacklistStr,
                           'whitelistStr': whitelistStr,
                           'enrollment_number': enrollment_number,
                           'first_beds': int(first_beds),
                           'second_beds': int(second_beds)
                           })


def execute_model(cm, county, county_provider_data, hold_capacity, model_df, to_add, white, turnover, ui):
    global enrollment_number
    global first_beds
    global second_beds
    # Build constraint matrix
    A = []
    # Bed number constraint
    A.append(model_df['BEDCERT'].values.T)
    sample_size = len(model_df)
    # Neighbor constraint
    for i in range(sample_size):
        tmp = model_df.iloc[i, 6:].values.astype('float64')
        tmp[i] = -1
        A.append(tmp)
    #  Build the upper bound (right side) of the inequality
    B = []
    # Bed number upper bound
    all_to_add = []
    for i in to_add:
        all_to_add.append(i)
    for i in white:
        all_to_add.append(i)
    added_index = county_provider_data[county_provider_data['PROVNUM'].isin(all_to_add)].index
    added_provider_beds = county_provider_data['BEDCERT'][added_index].sum()
    enrollment_number = enrollment_info[enrollment_info['COUNTY_NAME'] == county].reset_index()['ENROLLMENT'].sum()
    if first_beds == 0:
        first_beds = enrollment_number * cm * ui * turnover / 365
    else:
        second_beds = enrollment_number * cm * ui * turnover / 365
    enrollment = enrollment_number - added_provider_beds
    B.append((enrollment * cm * ui * turnover / 365) - hold_capacity)
    # Neighbor upper bound
    for i in range(sample_size):
        B.append(0)
    # Build the object function
    objective_func = model_df['COST'].values.T
    import cvxpy
    selection = cvxpy.Variable((sample_size), boolean=True, integer=True)
    total_cost = np.array(objective_func) * selection
    # constraints = np.array([a * selection <= B for a in np.array(A)])
    B = np.array(B).reshape(-1, 1)
    constraints = [A[i] * selection >= B[i] for i in range(len(model_df) + 1)]
    # constraints = feature_df['BEDCERT'].values.T * selection >= 5000
    network_optimization = cvxpy.Problem(cvxpy.Minimize(total_cost), constraints)
    network_optimization.solve(solver=cvxpy.GLPK_MI)
    # print(total_cost.value + costs)
    print('enrollment', enrollment)
    print('capacity constraints', enrollment * cm * ui * turnover / 365)

    if total_cost.value is None:
        selected_providers_index = []
    else:
        selected_providers_index = np.concatenate((model_df[selection.value == 1].index, added_index), axis=None)

    selected_providers = county_provider_data.iloc[selected_providers_index, :]
    print(selected_providers)
    return model_df, selected_providers, total_cost


def optimize(request):
    global cm_init
    global ui_init
    global turnover_init
    global min_rating_init
    global feature_1_init
    global feature_2_init
    global feature_3_init
    global county_name
    global first_beds
    global second_beds
    county = request.POST['county']
    cm = float(request.POST['cm'])
    ui = float(request.POST['ui']) / 100
    turnover = float(request.POST['turnover'])
    min_rating = int(request.POST['min_rating'])
    # Uncomment here
    # feature_1 = float(request.POST['feature_1'])
    # feature_2 = float(request.POST['feature_2'])
    # feature_3 = float(request.POST['feature_3'])
    to_delete = request.POST['to_delete']
    to_add = request.POST['to_add']
    to_delete_orig = to_delete
    to_add_orig = to_add
    to_delete = to_delete.replace(' ', '').split(';')
    to_add = to_add.replace(' ', '').split(';')
    company = request.POST['company']

    global feature_df
    global first_df
    global provider_info
    global first_result
    global second_result
    global first_cost
    global first_avg_score
    global second_cost
    global county_provider_data

    cm_init = request.POST['cm']
    ui_init = request.POST['ui']
    turnover_init = request.POST['turnover']
    min_rating_init = request.POST['min_rating']
    # Uncomment here
    # feature_1_init = feature_1
    # feature_2_init = feature_2
    # feature_3_init = feature_3

    global blacklistStr
    global whitelistStr

    blacklistStr = request.POST['blacklistStr']
    whitelistStr = request.POST['whitelistStr']
    black = []
    white = []

    if blacklistStr == '' and whitelistStr == '' and request.FILES:
        black, white = read_list_from_file(request.FILES['bwlist'], county)

    select_enrollment(company)

    # build the feature matrix
    filename = county + '.csv'
    distance_info = pd.read_csv('optimizer/static/distance/' + filename, encoding='ISO-8859-1')

    distance_info['origin_id'] = distance_info['origin_id'].astype('str')
    distance_info['destination_id'] = distance_info['destination_id'].astype('str')

    county_provider_data = provider_info[(provider_info['COUNTY_NAME'] == county)]

    county_provider_data = county_provider_data.reset_index(drop=True)
    county_provider_data = county_provider_data.rename(columns={'Total SNF Medicare Payment Amount': 'COST'})
    # debug
    # print(county_provider_data)

    distance_matrix = []
    for i in range(len(county_provider_data)):
        distance_arr = []
        original_id = str(county_provider_data.iloc[i, 0])
        # print("original", original_id)
        for j in range(len(county_provider_data)):
            if i == j:
                distance_arr.append(0.0)
                continue
            destination_id = str(county_provider_data.iloc[j, 0])
            distance_df = distance_info[((distance_info['origin_id'] == original_id) &
                                         (distance_info['destination_id'] == destination_id)) |
                                        ((distance_info['origin_id'] == destination_id) &
                                         (distance_info['destination_id'] == original_id))]
            # print(distance_df)
            if distance_df.reset_index()['DISTANCE'][0] > 20:
                distance_arr.append(0.0)
            else:
                distance_arr.append(1.0)
        distance_arr.append(original_id)
        distance_matrix.append(distance_arr)

    distance_matrix = np.array(distance_matrix)
    # distance_matrix.shape
    distance_df = pd.DataFrame(distance_matrix)
    distance_df.rename(columns={distance_df.columns[len(distance_df)]: "PROVNUM"}, inplace=True)
    county_provider_data['PROVNUM'] = county_provider_data['PROVNUM'].astype('str')
    all_feature_df = pd.merge(county_provider_data, distance_df, on=['PROVNUM'])

    distance_index = list(range(15, 15 + len(county_provider_data)))
    index = [2, 5, 9, 12, 13, 14]
    index = index + distance_index
    # print(index)
    # all_feature_df[list(range(1,60))] = all_feature_df[list(range(1,60))].astype('float')
    feature_df = all_feature_df.iloc[:, index]

    # all_feature_df[list(range(1, 60))] = all_feature_df[list(range(1, 60))].astype('float')

    model_df, costs_1, hold_capacity_1 = change_providers(feature_df, county_provider_data, to_delete, to_add)

    model_df, costs_2, hold_capacity_2 = change_providers(model_df, county_provider_data, black, white)
    # Overall rating constraint
    model_df = add_constraint(model_df, min_rating, 'OVERALL_RATING', True)
    # Make changes here.
    # 'FEATURE_X' should be replaced by the column name in the dataset.
    # model_df = add_constraint(model_df, feature_1, 'FEATURE_1', True)
    # model_df = add_constraint(model_df, feature_2, 'FEATURE_2', True)
    # model_df = add_constraint(model_df, feature_3, 'FEATURE_3', True)
    model_df, selected_providers, total_cost = execute_model(cm, county, county_provider_data,
                                                             hold_capacity_1 + hold_capacity_2,
                                                             model_df, to_add, white, turnover, ui)

    # cache to global variable

    first_df = selected_providers
    first_avg_score = first_df['OVERALL_RATING'].mean()
    if total_cost.value is None:
        first_result = False
        # feature_df = None
    else:
        first_cost = total_cost.value + costs_1 + costs_2
        feature_df = model_df
    print("first score", first_avg_score)
    print("second score", 0)
    print("first cost", first_cost)
    print("second cost", second_cost)
    print("first result", first_result)
    print("second result", second_result)
    county_name = county
    return render(request, 'second_model.html',
                  context={"first_df": first_df, "second_df": None, "first_avg_score": round(first_avg_score, 3),
                           "second_avg_score": 0.000, "first_cost": round(first_cost, 3),
                           "second_cost": round(second_cost, 3),
                           "first_result": first_result, "second_result": second_result, 'county': county,
                           'cm': cm_init,
                           'ui': ui_init, 'turnover': turnover_init, 'min_rating': min_rating_init,
                           'to_add': to_add_orig,
                           'to_delete': to_delete_orig,
                           # Uncomment here
                           # 'feature_1': feature_1_init,
                           # 'feature_2': feature_2_init,
                           # 'feature_3': feature_3_init,
                           'blacklistStr': blacklistStr,
                           'whitelistStr': whitelistStr,
                           'enrollment_number': enrollment_number,
                           'first_beds': int(first_beds),
                           'second_beds': int(second_beds)
                           })


def change_providers(df, county_data, to_delete, to_add):
    costs = 0
    hold_capacity = 0
    final_df = df.copy()
    #     Deal with to_delete first
    if to_delete is not None and len(to_delete) > 0:
        deleted_index = county_data[county_data['PROVNUM'].isin(to_delete)].index
        print(deleted_index)
        for i in deleted_index:
            if i in final_df.index:
                final_df = final_df.drop(i, axis=1).drop(i, axis=0)

    if to_add is not None and len(to_add) > 0:
        added_index = county_data[county_data['PROVNUM'].isin(to_add)].index
        added_index = list(set(added_index) & set(final_df.index))
        for i in added_index:
            if i in final_df.index:
                final_df = final_df.drop(i, axis=1).drop(i, axis=0)

        for i in added_index:
            costs += county_data['COST'][i]
            hold_capacity += county_data['BEDCERT'][i]
            print("cost", costs)
    # print('df.index', final_df.index)
    return final_df, costs, hold_capacity


def add_constraint(df, limit, header, larger_than):
    if larger_than:
        dropped = df[df[header] < limit].index
    else:
        dropped = df[df[header] > limit].index
    return df.drop(dropped, axis=1).drop(dropped, axis=0)


def read_list_from_file(path, county):
    if path is None:
        return [], []

    global blacklistStr
    global whitelistStr

    df_list = pd.read_csv(path)
    df_list = df_list[df_list['COUNTY'] == county]
    blacklist_df = df_list[df_list['FLAG'] == 'B']
    whitelist_df = df_list[df_list['FLAG'] == 'W']
    blacklist = blacklist_df['PROVNUM'].values.astype('str')
    whitelist = whitelist_df['PROVNUM'].values.astype('str')
    blacklistStr = ','.join(blacklist)
    whitelistStr = ','.join(whitelist)

    return blacklist, whitelist


def read_list_from_form():
    global blacklistStr
    global whitelistStr

    return blacklistStr.split(','), whitelistStr.split(',')


def select_enrollment(company):
    global enrollment_info

    if company == 'highmark':
        enrollment_info = enrollment_info[enrollment_info['COMPANY'] == 'Highmark']
    elif company == 'upmc':
        enrollment_info = enrollment_info[enrollment_info['COMPANY'] == 'UPMC']
