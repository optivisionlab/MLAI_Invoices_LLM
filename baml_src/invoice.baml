class Products{
    product_name string [] @alias("Danh sách tên sản phẩm") @description("List of names of products purchased")
    product_quantity string [] @alias("Danh sách số lượng sản phẩm") @description("List of quantities of each product purchased")
    product_unit_price string [] @alias("Danh sách đơn giá tiền sản phẩm") @description("List of unit prices of each product")
    product_price string [] @alias("Danh sách giá tiền sản phẩm") @description("List of prices of purchased products")
}

class Invoices {
    datetime string @alias("Thời gian mua hàng") @description("Order purchase time. Return format dd/mm/yyyy")
    products Products @alias("Danh sách sản phẩm") @description("List of products in invoice")
    total string @alias("Tổng tiền thanh toán") @description("Total amount of purchase invoice payment.") 
    @@alias("Thông tin giấy tờ") @@description("If no information found, leave blank")
}


function ExtractInvoices(images: image[]) -> Invoices {
  client "openai/gpt-4o" // Set OPENAI_API_KEY to use this client. openai/gpt-4o openai/o1 Gemini_2_0_pro Extract from this content:
  prompt #"
    
    {{_.role("user")}}
    Extract information about purchase time and amount paid.
    {% for img in images %}
      image-{{loop.index}}:  
      {{ img }}
    {% endfor %}
    {{ ctx.output_format }}
  "#
}