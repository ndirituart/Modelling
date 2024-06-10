def calculate_allocation(target_clicks, total_budget):
    # Click-through rates
    ctr_facebook = 15 / 100
    ctr_linkedin = 47 / 100
    ctr_instagram = 30 / 100
    ctr_whatsapp = 8 / 100

    # Calculate the impressions needed for each platform
    impressions_facebook = target_clicks / ctr_facebook
    impressions_linkedin = target_clicks / ctr_linkedin
    impressions_instagram = target_clicks / ctr_instagram
    impressions_whatsapp = target_clicks / ctr_whatsapp

    # Calculate the total impressions needed
    total_impressions = (impressions_facebook + impressions_linkedin + impressions_instagram + impressions_whatsapp)

    # Calculate the percentage of the total impressions for each platform
    percentage_facebook = impressions_facebook / total_impressions
    percentage_linkedin = impressions_linkedin / total_impressions
    percentage_instagram = impressions_instagram / total_impressions
    percentage_whatsapp = impressions_whatsapp / total_impressions

    # Allocate the budget based on the percentage of impressions needed from each platform
    budget_facebook = total_budget * percentage_facebook
    budget_linkedin = total_budget * percentage_linkedin
    budget_instagram = total_budget * percentage_instagram
    budget_whatsapp = total_budget * percentage_whatsapp

    # Create the results as a formatted string
    result_text = (
        f"To achieve {target_clicks} clicks, allocate your budget as follows:\n"
        f"Facebook: Ksh.{budget_facebook:.2f}\n"
        f"LinkedIn: Ksh.{budget_linkedin:.2f}\n"
        f"Instagram: Ksh.{budget_instagram:.2f}\n"
        f"WhatsApp: Ksh.{budget_whatsapp:.2f}"
    )

    return result_text

if __name__ == "__main__":
    # Get user input from the console
    target_clicks = int(input("Enter Target Clicks: "))
    total_budget = float(input("Enter Total Budget (Ksh): "))

    # Calculate the allocation and print the results
    allocation_result = calculate_allocation(target_clicks, total_budget)
    print(allocation_result)
