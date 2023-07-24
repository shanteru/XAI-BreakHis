lrp = LRP(model)
attributions = lrp.attribute(input_data, target=target_label)
