import featuretools as ft
import pandas as pd

# Creating a mock customer dataframe
customers_df = pd.DataFrame({
    'customer_id': [1, 2, 3, 4],
    'zipcode': [12345, 12346, 12347, 12348]
})

# Creating a mock sessions dataframe (imagine each session is a shopping session)
sessions_df = pd.DataFrame({
    'session_id': [1, 2, 3, 4],
    'customer_id': [1, 2, 1, 4],
    'session_start': pd.to_datetime(['2020-01-01 00:00:00', 
                                     '2020-01-02 00:00:00', 
                                     '2020-01-03 00:00:00', 
                                     '2020-01-04 00:00:00'])
})

# Creating an entity set
es = ft.EntitySet(id="customer_data")

# Adding the customer dataframe as an entity
es = es.add_dataframe(dataframe_name="customers",
                              dataframe=customers_df,
                              index="customer_id")

# Adding the sessions dataframe as an entity
es = es.add_dataframe(dataframe_name="sessions",
                              dataframe=sessions_df,
                              index="session_id",
                              time_index="session_start")

# Defining the relationship between customers and sessions
relationship = es.add_relationship(
  parent_dataframe_name='customers',
  parent_column_name='customer_id',
  child_dataframe_name='sessions',
  child_column_name='customer_id'
)

# Adding the relationship to the entity set
# es = es.add_relationship(relationship)

# Automatically generating features
features, feature_names = ft.dfs(entityset=es, 
                                 target_dataframe_name="customers",
                                 max_depth=2)

# Viewing the generated features
print(features)
print(feature_names)
