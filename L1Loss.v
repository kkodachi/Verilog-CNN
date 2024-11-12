module L1Loss #(
    parameter FC_OUTPUT_SIZE = 10
)(
    input clk,
    input rst,
    input start,
    input [31:0] predicted_probs [0:FC_OUTPUT_SIZE-1],  // Predicted values
    input [FC_OUTPUT_SIZE-1:0] ground_truth,  // Ground truth labels (one-hot encoded)
    output reg [31:0] loss,
    output reg done
);

    reg [31:0] sum_loss;
    reg [3:0] state;
    reg [9:0] i;
    localparam IDLE = 3'b000,
               INIT = 3'b001,
               WORK = 3'b010;

    always @(posedge clk) begin
        if (rst) begin
            state <= IDLE;
            sum_loss <= 0;
            loss <= 0;
            done <= 0;
            i <= 0;
        end else begin
            case (state)
                IDLE: begin
                    if (start) begin
                        state <= INIT;
                        sum_loss <= 0;
                        i <= 0;
                    end
                end

                INIT: begin
                    state <= WORK;
                end

                WORK: begin
                    if (ground_truth[i] == 1'b1) begin
                        sum_loss <= sum_loss + abs(predicted_probs[i] - 32'h3F800000);
                    end
                    if (i < FC_OUTPUT_SIZE - 1) begin
                        i <= i + 1;
                    end else begin
                        state <= IDLE;
                        loss <= sum_loss;
                        done <= 1;
                    end
                end
            endcase
        end
    end

    function [31:0] abs;
        input [31:0] value;
        begin
            if (value[31] == 1'b1) // if neg
                abs = ~value + 1;
            else
                abs = value;
        end
    endfunction

endmodule
