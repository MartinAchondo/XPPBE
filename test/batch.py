def solve_with_TFoptimizer(self, optimizer, N=1001, N_precond=10, batch_size=32):
    # ... (other parts of the function)

    # Convert the data meshes (X_r and XD) to TensorFlow constant tensors
    X_r_tensor = tf.constant(self.PINN_solver.mesh.X_r, dtype=self.DTYPE)
    XD_tensor = tf.constant(self.PINN_solver.mesh.XD_data, dtype=self.DTYPE)  # Replace with your actual XD data

    # Create TensorFlow Datasets from the data tensors
    full_dataset_X_r = tf.data.Dataset.from_tensor_slices(X_r_tensor)
    full_dataset_XD = tf.data.Dataset.from_tensor_slices(XD_tensor)

    # Apply batching to the datasets
    full_dataset_X_r = full_dataset_X_r.batch(batch_size)
    full_dataset_XD = full_dataset_XD.batch(batch_size)

    # Iterate through the training steps
    for i in pbar:
        # Shuffle the batches for each iteration
        shuffled_dataset_X_r = full_dataset_X_r.shuffle(buffer_size=len(self.PINN_solver.mesh.X_r), reshuffle_each_iteration=True)
        shuffled_dataset_XD = full_dataset_XD.shuffle(buffer_size=len(self.PINN_solver.mesh.XD_data), reshuffle_each_iteration=True)

        # Zip the two shuffled datasets together
        shuffled_combined_dataset = tf.data.Dataset.zip((shuffled_dataset_X_r, shuffled_dataset_XD))

        # Iterate through the shuffled batches
        for batch_X_r, batch_XD in shuffled_combined_dataset:
            # Call the train_step function for each batch
            loss, L_loss = train_step(batch_X_r, batch_XD, self.precondition)
            
            # Call the callback to update loss history
            self.callback(loss, L_loss)

            if self.iter > N_precond:
                self.precondition = False

            if self.iter % 10 == 0:
                pbar.set_description("Loss: {:6.4e}".format(self.current_loss))
    
    # ... (remaining parts of the function)
