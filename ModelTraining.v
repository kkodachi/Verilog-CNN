module ModelTraining #(
    parameter IMG_HEIGHT = 64,
    parameter IMG_WIDTH = 64,
    parameter KERNEL = 3,
    parameter CHANNELS = 1,
    parameter NEURONS = 30,
    parameter FC_INPUT_SIZE = 120,
    parameter FC_OUTPUT_SIZE = 10
)(
    input clk,
    input rst,
    input [7:0] img [0:IMG_WIDTH-1][0:IMG_HEIGHT-1],
    input [15:0] kernel [0:KERNEL-1][0:KERNEL-1],
    input start,
    input [FC_OUTPUT_SIZE-1:0] ground_truth,
    input [15:0] weights [0:FC_INPUT_SIZE-1][0:FC_OUTPUT_SIZE-1],
    input [15:0] bias [0:FC_OUTPUT_SIZE-1],
    output reg done,
    output reg [31:0] final_output [0:FC_OUTPUT_SIZE-1],
    output reg [31:0] loss
);

    reg [3:0] state;
    localparam  IDLE = 4'b0000,
            CONV2D_F = 4'b0001,    // Forward Convolution
            MAXPOOL_F = 4'b0010,   // Forward Max-Pooling
            FC_F = 4'b0011,        // Forward Fully Connected
            FC_B = 4'b0100,        // Backward Fully Connected
            MAXPOOL_B = 4'b0101,   // Backward Max-Pooling
            CONV2D_B = 4'b0110,    // Backward Convolution
            UPDATE_WEIGHTS = 4'b0111, // Update Weights
            DONE = 4'b1000,        // Done
            CALC_LOSS = 4'b1001;  

    reg start_conv2d_f,
        start_maxpool_f,
        start_fc_f,
        start_loss_calc,
        start_fc_b,
        start_maxpool_b,
        start_conv2d_b,
        start_ud;

    wire conv2d_f_done,
        maxpool_f_done,
        fc_f_done,
        loss_done,
        fc_b_done,
        maxpool_b_done,
        conv2d_b_done,
        ud_done;

    wire [15:0] featureMap [0:IMG_WIDTH-KERNEL][0:IMG_HEIGHT-KERNEL];
    wire [15:0] pooled_output [0:(IMG_WIDTH-KERNEL+1)/2-1][0:(IMG_HEIGHT-KERNEL+1)/2-1];
                
    Conv2d_Forward #(
        .IMG_HEIGHT(IMG_HEIGHT),
        .IMG_WIDTH(IMG_WIDTH),
        .KERNEL(KERNEL),
        .CHANNELS(CHANNELS),
        .NEURONS(NEURONS)
    ) conv2d_f (
        .clk(clk),
        .rst(rst),
        .start(start_conv2d_f),
        .img(img),
        .kernel(kernel),
        .featureMap(featureMap),
        .done(conv2d_f_done)
    );

    MaxPool_Forward #(
        .FM_height(IMG_HEIGHT-KERNEL+1),
        .FM_width(IMG_WIDTH-KERNEL+1),
        .KERNEL(2),
        .STRIDE(2)
    ) maxpool_f (
        .clk(clk),
        .rst(rst),
        .start(start_maxpool_f),
        .featureMap(featureMap),
        .pooled_output(pooled_output),
        .done(maxpool_f_done)
    );

    FullyConnect_Forward #(
        .input_size(FC_INPUT_SIZE),
        .output_size(FC_OUTPUT_SIZE)
    ) fc (
        .clk(clk),
        .rst(rst),
        .start(start_fc_f),
        .input_data(flattened_output),
        .weights(weights),
        .bias(bias),
        .output_data(final_output),
        .done(fc_f_done)
    );

    // TODO: check that connections between are correct
    L1Loss #(
        .FC_OUTPUT_SIZE(FC_OUTPUT_SIZE)
    ) l1_loss (
        .clk(clk),
        .rst(rst),
        .start(start_loss_calc),
        .predicted_probs(final_output),
        .ground_truth(ground_truth),
        .loss(loss),
        .done(loss_done)
    );

    FullyConnect_Backward #(
        .input_size(FC_INPUT_SIZE),
        .output_size(FC_OUTPUT_SIZE)
    ) fc_b (
        .clk(clk),
        .rst(rst),
        .start(start_fc_b),
        .input_data(flattened_output),
        .weights(weights),
        .bias(bias),
        .gradients(gradients_fc),
        .done(fc_b_done)
    );

    always @(posedge clk) begin
        if (rst) begin
            done <= 0;
            state <= IDLE;
            start_conv2d_f <= 0;
            start_maxpool_f <= 0;
            fc_f_done <= 0;
            loss_done <= 0;
            fc_b_done <= 0;
            maxpool_b_done <= 0;
            conv2d_b_done <= 0;
            ud_done <= 0;
        end else begin
            case (state)
                IDLE: begin
                    if (start) begin
                        state <= CONV2D_F;
                        start_conv2d_f <= 1;
                    end
                end

                CONV2D_F: begin
                    start_conv2d_f <= 0;
                    if (conv2d_f_done) begin
                        state <= MAXPOOL_F;
                        start_maxpool_f <= 1;
                    end
                end

                MAXPOOL_F: begin
                    start_maxpool_f <= 0;
                    if (maxpool_f_done) begin
                        state <= FC_F;
                        integer i, j, ind = 0;
                        for (i = 0; i < (IMG_WIDTH-KERNEL+1)/2; i = i + 1) begin
                            for (j = 0; j < (IMG_HEIGHT-KERNEL+1)/2; j = j + 1) begin
                                flattened_output[ind] <= pooled_output[i][j];
                                ind = ind + 1;
                            end
                        end
                    end
                end

                FC_F: begin
                    start_fc_f <= 0;
                    if (fc_f_done) begin
                        state <= CALC_LOSS;
                        start_loss_calc <= 1;
                    end
                end
                
                CALC_LOSS: begin
                    start_loss_calc <= 0;
                    if (loss_done) begin
                        state <= FC_B;
                        start_fc_b <= 1;
                    end
                end

                FC_B: begin
                    start_fc_b <= 0;
                    if (fc_b_done) begin
                        start_maxpool_b <= 1;
                    end
                end

                MAXPOOL_B: begin
                    start_maxpool_b <= 0;
                    if (maxpool_b_done) begin
                        start_conv2d_b <= 1;
                    end
                end

                CONV2D_B: begin
                    start_conv2d_b <= 0;
                    if (maxpool_b_done) begin
                        start_ud <= 1;
                    end
                end

                UPDATE_WEIGHTS begin
                    start_ud <= 0;
                end

                DONE: begin
                    state <= IDLE;
                end
            endcase
        end
    end

endmodule