module FullyConnect_Forward #(
    parameter input_size = 120,
    parameter output_size = 10
)(
    input clk,
    input rst,
    input start,
    input [15:0] input_data [0:input_size-1],
    input [15:0] weights [0:input_size-1][0:output_size-1], // matrix of weights
    input [15:0] bias [0:output_size-1],
    output reg [31:0] output_data [0:output_size-1], // fully connected output
    output reg done
);
    reg [9:0] i, j;
    reg [31:0] sum;
    reg [1:0] state;
    localparam IDLE = 2'b00, INIT = 2'b01, WORK = 2'b10;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            state <= IDLE;
            done <= 0;
        end else begin
            case (state)
                IDLE: begin
                    if (start) state <= INIT;
                end

                INIT: begin
                    state <= WORK;
                    i <= 0;
                    j <= 0;
                    sum <= 0;
                    done <= 0;
                end

                WORK: begin
                    sum <= sum + input_data[i] * weights[i][j];

                    if (i < output_size - 1) begin
                        i <= i + 1;
                    end else if (j < input_size - 1) begin
                        sum <= 0;
                        j <= j + 1;
                        i <= 0;
                        output_data[j] <= sum + bias[j];
                    end else begin
                        output_data[j] <= sum + bias[j];
                        state <= IDLE;
                        done <= 1;
                    end
                end
            endcase
        end
    end
endmodule

module FullyConnect_Backward #(
    parameter input_size = 120,
    parameter output_size = 10
)(
    input clk,
    input rst,
    input start,
    input [15:0] input_data [0:input_size-1],
    input [31:0] output_data [0:output_size-1], // fully connected output
    input [15:0] lossGrad_output [0:output_size-1],
    input [15:0] weights [0:input_size-1][0:output_size-1], // matrix of weights
    output reg [15:0] lossGrad_weights [0:output_size-1][0:input_size-1],
    output reg [15:0] lossGrad_bias [0:output_size-1],
    output reg [15:0] lossGrad_input [0:input_size-1], // Gradients for input
    output reg done
);
    reg [9:0] i, j, sum_i, sum_j;
    reg [31:0] sum;
    reg [1:0] state;
    localparam IDLE = 2'b00, INIT = 2'b01, WORK = 2'b10;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            done <= 0;
            state <= IDLE;
        end else begin
            case (state)
                IDLE: begin
                    if (start) begin
                        state <= INIT;
                    end
                end

                INIT: begin
                    i <= 0;
                    j <= 0;
                    sum <= 0;
                    sum_i <= 0;
                    sum_j <= 0;
                    state <= WORK;
                end

                WORK: begin
                    lossGrad_bias[i] <= lossGrad_output[i];
                    lossGrad_weights[j][i] <= lossGrad_output[i] * input_data[j];
                    sum = sum + lossGrad_output[sum_i] * weights[sum_j][sum_i];
                    
                    if (j < input_size - 1) begin
                        j <= j + 1;
                    end else if (i < output_size - 1) begin
                        j <= 0;
                        i <= i + 1;
                    end

                    if (sum_i < output_size - 1) begin
                        sum_i <= sum_i + 1;
                    end else if (sum_j < input_size - 1) begin
                        sum_i <= 0;
                        sum_j <= sum_j + 1;
                        lossGrad_input[sum_j] <= sum;
                        sum <= 0;
                    end else begin
                        lossGrad_input[sum_j] <= sum;
                        done <= 1;
                        state <= IDLE;
                    end
                end
            endcase
        end
    end
endmodule
