`timescale 1ns/1ps

module cnn_training_demo_tb();
    // Architecture Parameters
    parameter INPUT_WIDTH = 64;
    parameter INPUT_HEIGHT = 64;
    parameter INPUT_CHANNELS = 1;
    parameter WINDOW_SIZE = 3;
    parameter NUM_NEURONS = 30;
    parameter POOL_STRIDE = 2;
    parameter FC_OUTPUT_SIZE = 10;
    
    // Training Parameters
    parameter NUM_EPOCHS = 5;
    parameter BATCH_SIZE = 32;
    parameter NUM_BATCHES = 10;
    parameter NUM_TRAIN_SAMPLES = BATCH_SIZE * NUM_BATCHES;
    parameter FIXED_POINT_BITS = 16;
    parameter FRAC_BITS = 8;

    // Clock and Reset
    reg clk;
    reg reset;
    
    // Control Signals
    reg conv_enable, pool_enable, fc_enable;
    wire conv_done, pool_done, fc_done;
    
    // Training Data Storage
    reg [15:0] training_data [0:NUM_TRAIN_SAMPLES-1][0:INPUT_WIDTH*INPUT_HEIGHT*INPUT_CHANNELS-1];
    reg [FC_OUTPUT_SIZE-1:0] true_labels [0:NUM_TRAIN_SAMPLES-1];
    reg [15:0] network_outputs [0:FC_OUTPUT_SIZE-1];
    
    // Training Metrics
    reg [31:0] batch_loss;
    reg [31:0] epoch_loss;
    reg [31:0] total_loss;
    reg [31:0] epoch_loss_history [0:NUM_EPOCHS-1];
    reg [7:0] epoch_accuracy_history [0:NUM_EPOCHS-1];
    integer correct_predictions;
    integer total_samples;
    
    // Layer Outputs for Monitoring
    reg [15:0] conv_layer_output;
    reg [15:0] pool_layer_output;
    reg [15:0] fc_layer_output;
    
    // Training Progress Counters
    reg [3:0] current_epoch;
    reg [7:0] current_batch;
    reg [15:0] sample_counter;
    
    // Performance Monitoring
    time training_start_time;
    time training_end_time;
    reg [31:0] total_cycles;

    // Instance your original CNN modules
    conv2d #(
        .INPUT_WIDTH(INPUT_WIDTH),
        .INPUT_HEIGHT(INPUT_HEIGHT),
        .INPUT_CHANNELS(INPUT_CHANNELS),
        .WINDOW_SIZE(WINDOW_SIZE),
        .NUM_NEURONS(NUM_NEURONS)
    ) conv_layer (
        .clk(clk),
        .reset(reset),
        .enable(conv_enable),
        // ... other connections ...
    );

    max_pool #(
        .INPUT_WIDTH(INPUT_WIDTH-WINDOW_SIZE+1),
        .INPUT_HEIGHT(INPUT_HEIGHT-WINDOW_SIZE+1),
        .INPUT_CHANNELS(NUM_NEURONS),
        .STRIDE(POOL_STRIDE)
    ) pool_layer (
        .clk(clk),
        .reset(reset),
        .enable(pool_enable),
        // ... other connections ...
    );

    fully_connected #(
        .INPUT_SIZE((INPUT_WIDTH-WINDOW_SIZE+1)*(INPUT_HEIGHT-WINDOW_SIZE+1)*NUM_NEURONS/(POOL_STRIDE*POOL_STRIDE)),
        .OUTPUT_SIZE(FC_OUTPUT_SIZE)
    ) fc_layer (
        .clk(clk),
        .reset(reset),
        .enable(fc_enable),
        // ... other connections ...
    );

    // Clock Generation
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    // Print Architecture Details
    task print_architecture;
        begin
            $display("\nCNN Architecture Details:");
            $display("------------------------");
            $display("Input Layer: %0dx%0dx%0d", INPUT_WIDTH, INPUT_HEIGHT, INPUT_CHANNELS);
            $display("Convolution Layer: %0d filters of size %0dx%0d", NUM_NEURONS, WINDOW_SIZE, WINDOW_SIZE);
            $display("Pooling Layer: %0dx%0d stride", POOL_STRIDE, POOL_STRIDE);
            $display("Fully Connected Layer: %0d outputs", FC_OUTPUT_SIZE);
            $display("------------------------\n");
            
            $display("Training Configuration:");
            $display("------------------------");
            $display("Number of Epochs: %0d", NUM_EPOCHS);
            $display("Batch Size: %0d", BATCH_SIZE);
            $display("Number of Batches: %0d", NUM_BATCHES);
            $display("Total Training Samples: %0d", NUM_TRAIN_SAMPLES);
            $display("------------------------\n");
        end
    endtask

    // Initialize Network and Training Data
    task initialize_system;
        begin
            // Initialize control signals
            reset = 1;
            conv_enable = 0;
            pool_enable = 0;
            fc_enable = 0;
            
            // Initialize counters
            current_epoch = 0;
            current_batch = 0;
            sample_counter = 0;
            
            // Initialize metrics
            total_loss = 0;
            batch_loss = 0;
            correct_predictions = 0;
            total_samples = 0;
            total_cycles = 0;
            
            // Generate training data and labels
            generate_training_data();
            
            // Release reset
            #100 reset = 0;
        end
    endtask

    // Generate Training Data
    task generate_training_data;
        integer i, j;
        begin
            $display("Generating training data...");
            for (i = 0; i < NUM_TRAIN_SAMPLES; i = i + 1) begin
                // Generate input data
                for (j = 0; j < INPUT_WIDTH*INPUT_HEIGHT*INPUT_CHANNELS; j = j + 1) begin
                    training_data[i][j] = $random;
                end
                // Generate labels (one-hot encoded)
                true_labels[i] = 1 << ($random % FC_OUTPUT_SIZE);
            end
            $display("Training data generation complete\n");
        end
    endtask

    // Forward Pass
    task forward_pass;
        input integer sample_idx;
        begin
            // Convolution Layer
            conv_enable = 1;
            @(posedge conv_done);
            conv_enable = 0;
            conv_layer_output = conv_layer.feature_map[0][0][0];
            total_cycles = total_cycles + 1;

            // Pool Layer
            pool_enable = 1;
            @(posedge pool_done);
            pool_enable = 0;
            pool_layer_output = pool_layer.pooled_output[0][0][0];
            total_cycles = total_cycles + 1;

            // FC Layer
            fc_enable = 1;
            @(posedge fc_done);
            fc_enable = 0;
            fc_layer_output = fc_layer.output_data[0];
            total_cycles = total_cycles + 1;

            // Store outputs
            for (integer i = 0; i < FC_OUTPUT_SIZE; i = i + 1) begin
                network_outputs[i] = fc_layer.output_data[i];
            end
        end
    endtask

    // Calculate Loss and Update Metrics
    task calculate_metrics;
        input integer sample_idx;
        reg [31:0] temp_loss;
        reg [3:0] predicted_class;
        reg [3:0] true_class;
        begin
            predicted_class = get_prediction();
            true_class = get_true_label(sample_idx);
            
            if (predicted_class == true_class) begin
                correct_predictions = correct_predictions + 1;
            end
            total_samples = total_samples + 1;
            
            temp_loss = 0;
            for (integer i = 0; i < FC_OUTPUT_SIZE; i = i + 1) begin
                if (true_labels[sample_idx][i]) begin
                    temp_loss = temp_loss + (16'hFFFF - network_outputs[i]);
                end
            end
            batch_loss = batch_loss + temp_loss;
        end
    endtask

    // Helper Functions
    function [3:0] get_prediction;
        reg [15:0] max_val;
        reg [3:0] max_idx;
        begin
            max_val = network_outputs[0];
            max_idx = 0;
            for (integer i = 1; i < FC_OUTPUT_SIZE; i = i + 1) begin
                if (network_outputs[i] > max_val) begin
                    max_val = network_outputs[i];
                    max_idx = i;
                end
            end
            get_prediction = max_idx;
        end
    endfunction

    function [3:0] get_true_label;
        input integer sample_idx;
        begin
            for (integer i = 0; i < FC_OUTPUT_SIZE; i = i + 1) begin
                if (true_labels[sample_idx][i]) begin
                    get_true_label = i;
                    break;
                end
            end
        end
    endfunction

    // Print Training Summary
    task print_training_summary;
        integer i;
        real training_time_ms;
        begin
            training_time_ms = (training_end_time - training_start_time) / 1000000.0;
            
            $display("\nTraining Summary:");
            $display("------------------------");
            $display("Training Time: %.2f ms", training_time_ms);
            $display("Total Clock Cycles: %0d", total_cycles);
            $display("\nEpoch-wise Progress:");
            for (i = 0; i < NUM_EPOCHS; i = i + 1) begin
                $display("Epoch %0d - Loss: %f, Accuracy: %0d%%", 
                        i + 1,
                        $itor(epoch_loss_history[i]) / (1 << FRAC_BITS),
                        epoch_accuracy_history[i]);
            end
            $display("------------------------");
            $display("Final Accuracy: %0d%%\n", epoch_accuracy_history[NUM_EPOCHS-1]);
        end
    endtask

    // Main Training Process
    initial begin
        // Initialize and print architecture
        initialize_system();
        print_architecture();
        
        // Record start time
        training_start_time = $time;
        
        // Training loop
        for (current_epoch = 0; current_epoch < NUM_EPOCHS; current_epoch = current_epoch + 1) begin
            $display("\nStarting Epoch %0d", current_epoch + 1);
            epoch_loss = 0;
            
            for (current_batch = 0; current_batch < NUM_BATCHES; current_batch = current_batch + 1) begin
                batch_loss = 0;
                
                // Process each sample in batch
                for (sample_counter = 0; sample_counter < BATCH_SIZE; sample_counter = sample_counter + 1) begin
                    forward_pass(current_batch * BATCH_SIZE + sample_counter);
                    calculate_metrics(current_batch * BATCH_SIZE + sample_counter);
                end
                
                // Display batch results
                $display("Batch %0d/%0d - Loss: %f, Accuracy: %0d%%", 
                    current_batch + 1,
                    NUM_BATCHES,
                    $itor(batch_loss) / (BATCH_SIZE * (1 << FRAC_BITS)),
                    (correct_predictions * 100) / total_samples);
                
                epoch_loss = epoch_loss + batch_loss;
            end
            
            // Store epoch results
            epoch_loss_history[current_epoch] = epoch_loss / NUM_BATCHES;
            epoch_accuracy_history[current_epoch] = (correct_predictions * 100) / total_samples;
            
            // Display epoch results
            $display("\nEpoch %0d Results:", current_epoch + 1);
            $display("Average Loss: %f", $itor(epoch_loss_history[current_epoch]) / (1 << FRAC_BITS));
            $display("Accuracy: %0d%%", epoch_accuracy_history[current_epoch]);
        end
        
        // Record end time and print summary
        training_end_time = $time;
        print_training_summary();
        
        #100 $finish;
    end

    // Waveform Generation
    initial begin
        $dumpfile("cnn_training.vcd");
        $dumpvars(0, cnn_training_tb);
    end

endmodule