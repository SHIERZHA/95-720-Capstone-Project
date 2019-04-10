from django.shortcuts import render
import pandas as pd
import numpy as np

# Create your views here.

# Cache the processed data
first_df = None
# second_df = None
provider_info = None
enrollment_info = None
# contract_info = None
# enrolment_info = None
# provider_data = None
# contract_info_col_need = None
# contract_info_col_needed = None
# enrolment_info_col_needed = None
feature_df = None
county_provider_data = None
first_result = True
second_result = True
first_cost = 0
first_avg_score = 0
second_cost = 0


# enrollment = 0


def home(request):
    global provider_info
    global enrollment_info
    global feature_df
    global first_result
    global second_result

    feature_df = None
    # read file - takes a few seconds. Patience is appreciated
    if provider_info is None:
        provider_info = pd.read_csv('optimizer/static/dataset/provider_data.csv')
        enrollment_info = pd.read_csv('optimizer/static/dataset/enrollment_by_county.csv')
    return render(request, 'index.html', context={"first_result": first_result, "second_result": second_result})


def reset(request):
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

    return home(request)


def optimize(request):
    county = request.GET['county']
    cm = float(request.GET['cm'])
    ui = float(request.GET['ui'])
    turnover = float(request.GET['turnover'])
    min_rating = int(request.GET['min_rating'])
    to_delete = request.GET['to_delete']
    to_add = request.GET['to_add']
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
    second_df = None
    second_avg_score = 0
    first = False
    if feature_df is None:
        first = True
        # build the feature matrix
        filename = county + '.csv'
        distance_info = pd.read_csv('optimizer/static/distance/' + filename, encoding='ISO-8859-1')

        distance_info['origin_id'] = distance_info['origin_id'].astype('str')
        distance_info['destination_id'] = distance_info['destination_id'].astype('str')

        county_provider_data = provider_info[
            (provider_info['STATE'] == 'PA') & (provider_info['COUNTY_NAME'] == county)]

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
            # print(distance_arr)
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

    model_df, costs = change_providers(feature_df, county_provider_data, to_delete, to_add)
    # Overall rating constraint
    model_df = add_constraint(model_df, min_rating, 'OVERALL_RATING', True)
    # Build constraint matrix
    # min_rating = min_rating
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
    added_index = county_provider_data[county_provider_data['PROVNUM'].isin(to_add)].index
    added_provider_beds = county_provider_data['BEDCERT'][added_index].sum()
    enrollment = enrollment_info[enrollment_info['COUNTY_NAME'] == county].reset_index()['ENROLLMENT'][0] - added_provider_beds
    B.append((enrollment * cm * ui * turnover / 365))

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

    selected_providers_index = np.concatenate((model_df[selection.value == 1].index, added_index), axis=None)
    selected_providers = county_provider_data.iloc[selected_providers_index, :]
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
                           "second_avg_score": second_avg_score, "first_cost": first_cost, "second_cost": second_cost,
                           "first_result": first_result, "second_result": second_result, 'county': county, 'cm': cm,
                           'ui': ui, 'turnover': turnover, 'min_rating': min_rating, 'to_add': to_add_orig,
                           'to_delete': to_delete_orig})


def change_providers(df, county_data, to_delete, to_add):
    costs = 0
    final_df = df.copy()
    #     Deal with to_delete first
    if to_delete:
        deleted_index = county_data[county_data['PROVNUM'].isin(to_delete)].index
        final_df = df.drop(deleted_index, axis=1).drop(deleted_index, axis=0)

    if to_add:
        added_index = county_data[county_data['PROVNUM'].isin(to_add)].index
        added_index = list(set(added_index) & set(final_df.index))
        final_df = final_df.drop(added_index, axis=1).drop(added_index, axis=0)

        for i in added_index:
            costs += county_data['COST'][i]
            print("cost", costs)

    return final_df, costs


def add_constraint(df, limit, header, larger_than):
    if larger_than:
        dropped = df[df[header] < limit].index
    else:
        dropped = df[df[header] > limit].index
    return df.drop(dropped, axis=1).drop(dropped, axis=0)
