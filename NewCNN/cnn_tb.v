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
    
    // Memory interfaces for conv layer
    wire [15:0] conv_input_data;
    wire [15:0] conv_output_data;
    wire [$clog2(INPUT_WIDTH*INPUT_HEIGHT*INPUT_CHANNELS)-1:0] conv_input_addr;
    wire [$clog2((INPUT_WIDTH-WINDOW_SIZE+1)*(INPUT_HEIGHT-WINDOW_SIZE+1)*NUM_NEURONS)-1:0] conv_output_addr;
    reg conv_input_valid;
    wire conv_output_valid;

    // Memory interfaces for pool layer
    wire [15:0] pool_input_data;
    wire [15:0] pool_output_data;
    wire pool_output_valid;

    // Memory interfaces for FC layer
    wire [15:0] fc_input_data;
    wire [15:0] fc_output_data;
    wire fc_output_valid;
    
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
    
    // Loop variables
    integer i, j, k;

    // Instance your original CNN modules with proper port connections
    conv2d #(
        .INPUT_WIDTH(INPUT_WIDTH),
        .INPUT_HEIGHT(INPUT_HEIGHT),
        .INPUT_CHANNELS(INPUT_CHANNELS),
        .WINDOW_SIZE(WINDOW_SIZE),
        .NUM_NEURONS(NUM_NEURONS)
    ) conv_layer_inst (
        .clk(clk),
        .reset(reset),
        .enable(conv_enable),
        .input_data(conv_input_data),
        .input_addr(conv_input_addr),
        .input_valid(conv_input_valid),
        .feature_map(conv_output_data),
        .output_addr(conv_output_addr),
        .output_valid(conv_output_valid),
        .conv_done(conv_done)
    );

    max_pool #(
        .INPUT_WIDTH(INPUT_WIDTH-WINDOW_SIZE+1),
        .INPUT_HEIGHT(INPUT_HEIGHT-WINDOW_SIZE+1),
        .INPUT_CHANNELS(NUM_NEURONS),
        .STRIDE(POOL_STRIDE)
    ) pool_layer_inst (
        .clk(clk),
        .reset(reset),
        .enable(pool_enable),
        .input_data(pool_input_data),
        .input_valid(conv_output_valid),
        .pooled_output(pool_output_data),
        .output_valid(pool_output_valid),
        .pool_done(pool_done)
    );

    fully_connected #(
        .INPUT_SIZE((INPUT_WIDTH-WINDOW_SIZE+1)*(INPUT_HEIGHT-WINDOW_SIZE+1)*NUM_NEURONS/(POOL_STRIDE*POOL_STRIDE)),
        .OUTPUT_SIZE(FC_OUTPUT_SIZE)
    ) fc_layer_inst (
        .clk(clk),
        .reset(reset),
        .enable(fc_enable),
        .input_data(fc_input_data),
        .input_valid(pool_output_valid),
        .output_data(fc_output_data),
        .output_valid(fc_output_valid),
        .fc_done(fc_done)
    );

    // Clock Generation
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    // Initialize system
    initial begin
        // Initialize control signals
        reset = 1;
        conv_enable = 0;
        pool_enable = 0;
        fc_enable = 0;
        conv_input_valid = 0;
        
        // Initialize counters
        current_epoch = 0;
        current_batch = 0;
        sample_counter = 0;
        
        // Initialize metrics
        total_loss = 0;
        batch_loss = 0;
        correct_predictions = 0;
        total_samples = 0;
        
        // Generate training data
        for (i = 0; i < NUM_TRAIN_SAMPLES; i = i + 1) begin
            for (j = 0; j < INPUT_WIDTH*INPUT_HEIGHT*INPUT_CHANNELS; j = j + 1) begin
                training_data[i][j] = $random;
            end
            true_labels[i] = 1 << ($random % FC_OUTPUT_SIZE);
        end
        
        // Print architecture details
        $display("\nCNN Architecture Details:");
        $display("------------------------");
        $display("Input Layer: %0dx%0dx%0d", INPUT_WIDTH, INPUT_HEIGHT, INPUT_CHANNELS);
        $display("Convolution Layer: %0d filters of size %0dx%0d", NUM_NEURONS, WINDOW_SIZE, WINDOW_SIZE);
        $display("Pooling Layer: %0dx%0d stride", POOL_STRIDE, POOL_STRIDE);
        $display("Fully Connected Layer: %0d outputs", FC_OUTPUT_SIZE);
        $display("------------------------\n");
        
        // Release reset after 100ns
        #100 reset = 0;
        conv_input_valid = 1;
        
        // Start training
        for (current_epoch = 0; current_epoch < NUM_EPOCHS; current_epoch = current_epoch + 1) begin
            $display("\nStarting Epoch %0d", current_epoch + 1);
            epoch_loss = 0;
            
            for (current_batch = 0; current_batch < NUM_BATCHES; current_batch = current_batch + 1) begin
                batch_loss = 0;
                
                // Process each sample in batch
                for (sample_counter = 0; sample_counter < BATCH_SIZE; sample_counter = sample_counter + 1) begin
                    // Forward pass
                    conv_enable = 1;
                    @(posedge conv_done) conv_enable = 0;
                    
                    pool_enable = 1;
                    @(posedge pool_done) pool_enable = 0;
                    
                    fc_enable = 1;
                    @(posedge fc_done) fc_enable = 0;
                    
                    // Store outputs
                    conv_layer_output = conv_output_data;
                    pool_layer_output = pool_output_data;
                    fc_layer_output = fc_output_data;
                    
                    // Update metrics
                    batch_loss = batch_loss + calculate_sample_loss(current_batch * BATCH_SIZE + sample_counter);
                end
                
                // Display batch results
                $display("Batch %0d/%0d - Loss: %f, Accuracy: %0d%%", 
                    current_batch + 1,
                    NUM_BATCHES,
                    real'(batch_loss) / (BATCH_SIZE * (1 << FRAC_BITS)),
                    (correct_predictions * 100) / total_samples);
                
                epoch_loss = epoch_loss + batch_loss;
            end
            
            // Store epoch results
            epoch_loss_history[current_epoch] = epoch_loss / NUM_BATCHES;
            epoch_accuracy_history[current_epoch] = (correct_predictions * 100) / total_samples;
            
            // Display epoch results
            $display("\nEpoch %0d Results:", current_epoch + 1);
            $display("Average Loss: %f", real'(epoch_loss_history[current_epoch]) / (1 << FRAC_BITS));
            $display("Accuracy: %0d%%", epoch_accuracy_history[current_epoch]);
        end
        
        // Display final results
        $display("\nTraining Complete!");
        $display("Final Accuracy: %0d%%\n", epoch_accuracy_history[NUM_EPOCHS-1]);
        
        #100 $finish;
    end

    // Function to calculate loss for a single sample
    function [31:0] calculate_sample_loss;
        input integer sample_idx;
        reg [31:0] temp_loss;
    begin
        temp_loss = 0;
        for (i = 0; i < FC_OUTPUT_SIZE; i = i + 1) begin
            if (true_labels[sample_idx][i]) begin
                temp_loss = temp_loss + (16'hFFFF - fc_output_data);
            end
        end
        calculate_sample_loss = temp_loss;
    end
    endfunction

    // Generate waveform file
    initial begin
        $dumpfile("cnn_training.vcd");
        $dumpvars(0, cnn_training_demo_tb);
    end

endmodule