import os
import requests
from datetime import datetime
import locale
import geocoder

class TravelCostCalculator:
    """
    Calculate estimated travel costs using LLM and real-time exchange rates.
    """
    
    def __init__(self):
        self.exchange_api_key = os.getenv("EXCHANGERATE_API_KEY")  # Optional - uses free API if not provided
        
    def detect_user_currency(self):
        """
        Detect user's currency based on their location.
        Falls back to INR if detection fails.
        """
        try:
            # Try to detect user's location
            g = geocoder.ip('me')
            if g.ok and g.country:
                country_code = g.country
                currency_map = {
                    'IN': 'INR',
                    'US': 'USD',
                    'GB': 'GBP',
                    'EU': 'EUR',
                    'AU': 'AUD',
                    'CA': 'CAD',
                    'JP': 'JPY',
                    'CN': 'CNY',
                    'SG': 'SGD',
                    'AE': 'AED',
                    'SA': 'SAR',
                    'MY': 'MYR',
                    'TH': 'THB',
                    'ID': 'IDR',
                }
                return currency_map.get(country_code, 'INR')
        except:
            pass
        
        # Fallback to system locale
        try:
            loc = locale.getdefaultlocale()
            if loc and loc[0]:
                country_code = loc[0].split('_')[-1]
                if country_code == 'IN':
                    return 'INR'
                elif country_code == 'US':
                    return 'USD'
        except:
            pass
        
        return 'INR'  # Default fallback
    
    def get_destination_currency(self, destination):
        """
        Get currency code for destination country.
        """
        # Common destination to currency mapping
        currency_map = {
            'india': 'INR',
            'usa': 'USD',
            'united states': 'USD',
            'uk': 'GBP',
            'united kingdom': 'GBP',
            'bali': 'IDR',
            'indonesia': 'IDR',
            'thailand': 'THB',
            'bangkok': 'THB',
            'singapore': 'SGD',
            'malaysia': 'MYR',
            'dubai': 'AED',
            'uae': 'AED',
            'japan': 'JPY',
            'tokyo': 'JPY',
            'china': 'CNY',
            'france': 'EUR',
            'paris': 'EUR',
            'germany': 'EUR',
            'italy': 'EUR',
            'spain': 'EUR',
            'australia': 'AUD',
            'canada': 'CAD',
            'mexico': 'MXN',
            'brazil': 'BRL',
            'switzerland': 'CHF',
            'south korea': 'KRW',
            'hong kong': 'HKD',
            'new zealand': 'NZD',
            'vietnam': 'VND',
            'turkey': 'TRY',
            'egypt': 'EGP',
            'south africa': 'ZAR',
            'maldives': 'MVR',
            'sri lanka': 'LKR',
            'nepal': 'NPR',
            'bhutan': 'BTN',
        }
        
        dest_lower = destination.lower()
        for key, currency in currency_map.items():
            if key in dest_lower:
                return currency
        
        return 'USD'  # Default to USD if not found
    
    def get_exchange_rate(self, from_currency, to_currency):
        """
        Get exchange rate between two currencies using a free API.
        """
        if from_currency == to_currency:
            return 1.0
        
        try:
            # Using exchangerate-api.com (free tier: 1,500 requests/month)
            url = f"https://api.exchangerate-api.com/v4/latest/{from_currency}"
            response = requests.get(url, timeout=5, verify=False)
            data = response.json()
            
            if 'rates' in data and to_currency in data['rates']:
                return data['rates'][to_currency]
        except Exception as e:
            print(f"Exchange rate fetch failed: {e}")
        
        # Fallback to approximate rates if API fails
        fallback_rates = {
            ('USD', 'INR'): 83.0,
            ('INR', 'USD'): 0.012,
            ('USD', 'EUR'): 0.92,
            ('EUR', 'USD'): 1.09,
            ('USD', 'GBP'): 0.79,
            ('GBP', 'USD'): 1.27,
            ('USD', 'IDR'): 15600,
            ('IDR', 'USD'): 0.000064,
            ('INR', 'IDR'): 188,
            ('IDR', 'INR'): 0.0053,
        }
        
        return fallback_rates.get((from_currency, to_currency), 1.0)
    
    def calculate_cost_with_llm(self, itinerary_text, destination, num_days, budget_level, llm):
        """
        Use LLM to analyze itinerary and estimate costs.
        Returns cost breakdown in USD.
        """
        from langchain.prompts import PromptTemplate
        
        prompt = PromptTemplate.from_template(
            """
You are a travel cost estimation expert. Analyze the following itinerary and provide a detailed cost breakdown for 1 person.

Destination: {destination}
Trip Duration: {num_days} days
Budget Level: {budget_level}

Itinerary:
{itinerary}

Provide a realistic cost estimate in USD with the following breakdown:
1. Accommodation (per night average × {num_days} nights)
2. Food & Dining (per day average × {num_days} days)
3. Activities & Attractions (total for all activities mentioned)
4. Local Transportation (taxis, metro, buses for {num_days} days)
5. Shopping & Miscellaneous (estimated)

Format your response EXACTLY as follows (use only numbers, no currency symbols):
ACCOMMODATION: [amount]
FOOD: [amount]
ACTIVITIES: [amount]
TRANSPORTATION: [amount]
MISCELLANEOUS: [amount]

Be realistic based on the {budget_level} budget level:
- Low: Budget hostels, street food, free/cheap activities
- Medium: 3-star hotels, mid-range restaurants, popular attractions
- High: 4-5 star hotels, fine dining, premium experiences
"""
        )
        
        formatted_prompt = prompt.format(
            destination=destination,
            num_days=num_days,
            budget_level=budget_level,
            itinerary=itinerary_text[:3000]  # Limit to avoid token limits
        )
        
        try:
            response = llm.invoke(formatted_prompt)
            return self._parse_cost_response(response)
        except Exception as e:
            print(f"LLM cost calculation failed: {e}")
            # Fallback to rule-based estimation
            return self._fallback_cost_estimation(num_days, budget_level)
    
    def _parse_cost_response(self, response):
        """
        Parse LLM response to extract cost breakdown.
        """
        import re
        
        costs = {
            'accommodation': 0,
            'food': 0,
            'activities': 0,
            'transportation': 0,
            'miscellaneous': 0
        }
        
        patterns = {
            'accommodation': r'ACCOMMODATION:\s*(\d+(?:\.\d+)?)',
            'food': r'FOOD:\s*(\d+(?:\.\d+)?)',
            'activities': r'ACTIVITIES:\s*(\d+(?:\.\d+)?)',
            'transportation': r'TRANSPORTATION:\s*(\d+(?:\.\d+)?)',
            'miscellaneous': r'MISCELLANEOUS:\s*(\d+(?:\.\d+)?)'
        }
        
        for category, pattern in patterns.items():
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                costs[category] = float(match.group(1))
        
        # If parsing failed, use fallback
        if sum(costs.values()) == 0:
            return None
        
        return costs
    
    def _fallback_cost_estimation(self, num_days, budget_level):
        """
        Rule-based cost estimation as fallback.
        """
        budget_multipliers = {
            'Low': {'accommodation': 25, 'food': 15, 'activities': 20, 'transport': 10, 'misc': 10},
            'Medium': {'accommodation': 70, 'food': 40, 'activities': 50, 'transport': 20, 'misc': 30},
            'High': {'accommodation': 200, 'food': 100, 'activities': 150, 'transport': 40, 'misc': 60}
        }
        
        multiplier = budget_multipliers.get(budget_level, budget_multipliers['Medium'])
        
        return {
            'accommodation': multiplier['accommodation'] * num_days,
            'food': multiplier['food'] * num_days,
            'activities': multiplier['activities'] * num_days,
            'transportation': multiplier['transport'] * num_days,
            'miscellaneous': multiplier['misc'] * num_days
        }
    
    def format_currency(self, amount, currency):
        """
        Format amount with appropriate currency symbol and formatting.
        """
        symbols = {
            'INR': '₹',
            'USD': '$',
            'EUR': '€',
            'GBP': '£',
            'JPY': '¥',
            'AUD': 'A$',
            'CAD': 'C$',
            'SGD': 'S$',
            'AED': 'AED ',
            'IDR': 'Rp ',
            'THB': '฿',
            'MYR': 'RM ',
        }
        
        symbol = symbols.get(currency, f'{currency} ')
        
        # Format with commas and 2 decimal places
        if currency in ['JPY', 'IDR', 'VND']:  # Currencies without decimals
            return f"{symbol}{amount:,.0f}"
        else:
            return f"{symbol}{amount:,.2f}"
    
    def calculate_trip_cost(self, itinerary_text, destination, num_days, budget_level, llm):
        """
        Main function to calculate and return trip cost in multiple currencies.
        """
        # Detect currencies
        user_currency = self.detect_user_currency()
        dest_currency = self.get_destination_currency(destination)
        
        # Get cost breakdown in USD
        cost_breakdown = self.calculate_cost_with_llm(
            itinerary_text, destination, num_days, budget_level, llm
        )
        
        if not cost_breakdown:
            cost_breakdown = self._fallback_cost_estimation(num_days, budget_level)
        
        # Calculate total in USD
        total_usd = sum(cost_breakdown.values())
        
        # Convert to user currency and destination currency
        user_rate = self.get_exchange_rate('USD', user_currency)
        dest_rate = self.get_exchange_rate('USD', dest_currency)
        
        total_user_currency = total_usd * user_rate
        total_dest_currency = total_usd * dest_rate
        
        # Convert breakdown to user currency
        breakdown_user_currency = {
            category: amount * user_rate 
            for category, amount in cost_breakdown.items()
        }
        
        return {
            'user_currency': user_currency,
            'dest_currency': dest_currency,
            'total_user_currency': total_user_currency,
            'total_dest_currency': total_dest_currency,
            'breakdown_user_currency': breakdown_user_currency,
            'breakdown_usd': cost_breakdown
        }