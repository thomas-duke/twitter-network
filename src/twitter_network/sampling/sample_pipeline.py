####################################################################################
# 02: Sampling
####################################################################################
sample = Pipeline(
    [
        # Hide random edges in the undirected graph
        # The number of hidden edges is N_hidden, as defined in the parameters file
        node(
            hide_edges,
            inputs=dict(G="G_train", edges="edges_all", parameters="parameters"),
            outputs=dict(subG="subG_train", hidden="edges_hidden"),
            tags=["hide_edges"],
        ),
        # Generate fake edges
        # The number of fake edges is N_fake, as defined in the parameters file
        node(
            fake_edges,
            inputs=dict(
                targets="adj_train_full",
                subG="subG_train",
                hidden="edges_hidden",
                test="edges_test",
                parameters="parameters",
            ),
            outputs="edges_fake",
            tags=["fake_edges"],
        ),
        # Create a sample dataframe from the hidden and fake edges
        node(
            create_sample,
            inputs=dict(hidden="edges_hidden", fakes="edges_fake"),
            outputs="sample_50",
            tags=["create_sample"],
        ),
        # Split the sample into training and validation sets
        # The proportion of examples used for validation is valid_size, as defined
        # in the parameters file
        node(
            split_sample,
            inputs=dict(sample="sample_50", parameters="parameters"),
            outputs=["sample_train", "sample_valid"],
            tags=["split_sample"],
        ),
        # Extract class labels and edges from training set
        node(
            extract_classes,
            inputs="sample_train",
            outputs=dict(edges="edges_train", classes="y_train"),
            tags=["extract_classes"],
        ),
        # Extract class labels and edges from validation set
        node(
            extract_classes,
            inputs="sample_valid",
            outputs=dict(edges="edges_valid", classes="y_valid"),
            tags=["extract_classes"],
        ),
    ],
    name="sample",
)
