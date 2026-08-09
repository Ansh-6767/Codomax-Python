import re
import random
import math
from urllib.parse import urlparse

# ============================================================
# PHISHGUARD - PHISHING URL DETECTOR
# Single-file Python project
# ============================================================

try:
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
except ImportError:
    print("ERROR: This compiler does not have required libraries.")
    print("Required: pandas, scikit-learn")
    print("Try Google Colab, Kaggle, Replit, or another compiler supporting")
    print("Python packages.")
    raise SystemExit


# ============================================================
# 1. SUSPICIOUS KEYWORDS
# ============================================================

SUSPICIOUS_WORDS = [
    "login",
    "signin",
    "verify",
    "verification",
    "secure",
    "account",
    "update",
    "password",
    "bank",
    "wallet",
    "payment",
    "confirm",
    "recover",
    "unlock",
    "security",
    "bonus",
    "free",
    "gift",
    "urgent"
]

URL_SHORTENERS = [
    "bit.ly",
    "tinyurl.com",
    "goo.gl",
    "t.co",
    "ow.ly",
    "is.gd",
    "cutt.ly"
]


# ============================================================
# 2. IP ADDRESS DETECTOR
# ============================================================

def has_ip_address(url):

    pattern = r"^(?:https?://)?(?:\d{1,3}\.){3}\d{1,3}"

    return int(bool(re.search(pattern, url)))


# ============================================================
# 3. FEATURE EXTRACTION
# ============================================================

def extract_features(url):

    original_url = url

    # Add protocol if missing
    if not re.match(r"^https?://", url, re.IGNORECASE):
        url = "http://" + url

    parsed = urlparse(url)

    hostname = parsed.hostname or ""

    # Remove www.
    clean_hostname = hostname.lower().replace("www.", "")

    # Extract domain
    domain_parts = clean_hostname.split(".")

    if len(domain_parts) >= 2:
        domain = domain_parts[-2]
    else:
        domain = clean_hostname

    subdomain_count = max(0, len(domain_parts) - 2)

    url_lower = url.lower()

    # Count suspicious words
    suspicious_word_count = sum(
        1 for word in SUSPICIOUS_WORDS
        if word in url_lower
    )

    # Count digits
    digit_count = sum(
        1 for character in url
        if character.isdigit()
    )

    # Count special characters
    special_count = len(
        re.findall(r"[^a-zA-Z0-9]", url)
    )

    features = {

        "url_length":
            len(original_url),

        "domain_length":
            len(domain),

        "path_length":
            len(parsed.path),

        "query_length":
            len(parsed.query),

        "num_dots":
            original_url.count("."),

        "num_hyphens":
            original_url.count("-"),

        "num_slashes":
            original_url.count("/"),

        "num_at":
            original_url.count("@"),

        "num_question":
            original_url.count("?"),

        "num_equal":
            original_url.count("="),

        "num_ampersand":
            original_url.count("&"),

        "num_percent":
            original_url.count("%"),

        "num_digits":
            digit_count,

        "num_special":
            special_count,

        "has_ip":
            has_ip_address(original_url),

        "uses_https":
            int(parsed.scheme.lower() == "https"),

        "subdomain_count":
            subdomain_count,

        "suspicious_word_count":
            suspicious_word_count,

        "has_suspicious_words":
            int(suspicious_word_count > 0),

        "has_shortener":
            int(
                any(
                    service in url_lower
                    for service in URL_SHORTENERS
                )
            ),

        "has_port":
            int(parsed.port is not None)
            if parsed.port else 0
    }

    return features


# ============================================================
# 4. TRAINING DATA
# ============================================================

LEGITIMATE_DOMAINS = [
    "google.com",
    "microsoft.com",
    "apple.com",
    "amazon.com",
    "github.com",
    "linkedin.com",
    "wikipedia.org",
    "python.org",
    "mozilla.org",
    "nasa.gov",
    "openai.com",
    "ibm.com",
    "adobe.com",
    "netflix.com",
    "spotify.com",
    "stackoverflow.com",
    "reddit.com",
    "cloudflare.com",
    "ubuntu.com",
    "mit.edu"
]

PHISHING_DOMAINS = [
    "secure-login-verify.com",
    "account-security-update.com",
    "paypal-login-confirm.com",
    "bank-verification-alert.com",
    "amazon-account-security.com",
    "microsoft-account-verify.com",
    "free-gift-card-login.com",
    "password-reset-security.com",
    "wallet-payment-confirm.com",
    "verify-account-now.com",
    "secure-bank-login.com",
    "account-unlock-security.com",
    "confirm-your-payment.com",
    "urgent-account-verification.com",
    "security-check-login.com"
]


# ============================================================
# 5. URL GENERATORS
# ============================================================

def generate_legitimate_url():

    domain = random.choice(LEGITIMATE_DOMAINS)

    paths = [
        "",
        "/",
        "/about",
        "/products",
        "/services",
        "/contact",
        "/blog",
        "/news",
        "/docs",
        "/support",
        "/download",
        "/research"
    ]

    path = random.choice(paths)

    return "https://www." + domain + path


def generate_phishing_url():

    domain = random.choice(PHISHING_DOMAINS)

    patterns = [

        f"http://{domain}/login",

        f"http://{domain}/verify-account",

        f"http://{domain}/secure/login",

        f"http://{domain}/account/update",

        f"http://{domain}/password/reset",

        f"http://{domain}/confirm?user=12345",

        f"http://{domain}/login?verify=true",

        f"http://{domain}/security/check",

        f"http://{domain}/payment/confirm"

    ]

    return random.choice(patterns)


# ============================================================
# 6. CREATE TRAINING DATA
# ============================================================

def create_dataset():

    urls = []
    labels = []

    # Legitimate
    for _ in range(1500):

        url = generate_legitimate_url()

        urls.append(url)

        # 0 = legitimate
        labels.append(0)

    # Phishing
    for _ in range(1500):

        url = generate_phishing_url()

        urls.append(url)

        # 1 = phishing
        labels.append(1)

    feature_rows = []

    for url in urls:

        feature_rows.append(
            extract_features(url)
        )

    X = pd.DataFrame(feature_rows)

    y = pd.Series(labels)

    return X, y


# ============================================================
# 7. TRAIN MODEL
# ============================================================

print("\n" + "=" * 60)
print("        PHISHGUARD - PHISHING URL DETECTOR")
print("=" * 60)

print("\nPreparing training data...")

X, y = create_dataset()

print("Dataset created:", len(X), "URLs")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training model...")

model = RandomForestClassifier(
    n_estimators=150,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

print("Model trained successfully.")


# ============================================================
# 8. MODEL EVALUATION
# ============================================================

predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\n" + "-" * 60)
print("MODEL PERFORMANCE")
print("-" * 60)

print(
    f"Accuracy: {accuracy * 100:.2f}%"
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        target_names=[
            "Legitimate",
            "Phishing"
        ]
    )
)


# ============================================================
# 9. URL ANALYZER
# ============================================================

def analyze_url(url):

    features = extract_features(url)

    feature_df = pd.DataFrame(
        [features]
    )

    prediction = model.predict(
        feature_df
    )[0]

    probabilities = model.predict_proba(
        feature_df
    )[0]

    phishing_probability = (
        probabilities[1] * 100
    )

    return (
        prediction,
        phishing_probability,
        features
    )


# ============================================================
# 10. EXPLANATION SYSTEM
# ============================================================

def explain_url(features):

    reasons = []

    if features["has_ip"]:

        reasons.append(
            "Uses an IP address instead of a normal domain."
        )

    if features["has_suspicious_words"]:

        reasons.append(
            "Contains suspicious keywords related to "
            "login, verification, payment or security."
        )

    if features["url_length"] > 100:

        reasons.append(
            "URL is unusually long."
        )

    if features["num_hyphens"] >= 3:

        reasons.append(
            "Domain contains many hyphens."
        )

    if features["num_at"] > 0:

        reasons.append(
            "Contains the '@' character."
        )

    if features["subdomain_count"] >= 2:

        reasons.append(
            "Contains multiple subdomains."
        )

    if features["has_shortener"]:

        reasons.append(
            "Uses a URL shortening service."
        )

    if not features["uses_https"]:

        reasons.append(
            "Does not use HTTPS."
        )

    if features["num_percent"] >= 3:

        reasons.append(
            "Contains multiple encoded characters."
        )

    if not reasons:

        reasons.append(
            "No major suspicious structural indicators detected."
        )

    return reasons


# ============================================================
# 11. INTERACTIVE URL SCANNER
# ============================================================

while True:

    print("\n" + "=" * 60)

    url = input(
        "\nEnter URL to analyze "
        "(or type 'exit' to quit): "
    )

    if url.lower().strip() == "exit":

        print("\nThank you for using PhishGuard.")
        break

    if not url.strip():

        print("\nPlease enter a URL.")

        continue

    prediction, probability, features = analyze_url(url)

    print("\n" + "-" * 60)
    print("ANALYSIS RESULT")
    print("-" * 60)

    print("\nURL:")
    print(url)

    print(
        "\nPhishing Probability:"
        f" {probability:.2f}%"
    )

print("\n" + "=" * 60)
print("FINAL VERDICT")
print("=" * 60)

if prediction == 1:

    confidence = probability

    print(
        f"\n⚠️ YES — IT'S MOST LIKELY PHISHING"
    )

    print(
        f"Confidence: {confidence:.2f}%"
    )

else:

    confidence = 100 - probability

    print(
        f"\n✅ YES — IT'S MOST LIKELY LEGITIMATE"
    )

    print(
        f"Confidence: {confidence:.2f}%"
    )

    print("\nWhy?")

    reasons = explain_url(features)

    for reason in reasons:

        print(" •", reason)

    print("\nFeature Analysis:")

    print(
        f" • URL length: "
        f"{features['url_length']}"
    )

    print(
        f" • Domain length: "
        f"{features['domain_length']}"
    )

    print(
        f" • Number of dots: "
        f"{features['num_dots']}"
    )

    print(
        f" • Number of hyphens: "
        f"{features['num_hyphens']}"
    )

    print(
        f" • Number of digits: "
        f"{features['num_digits']}"
    )

    print(
        f" • IP address: "
        f"{'Yes' if features['has_ip'] else 'No'}"
    )

    print(
        f" • HTTPS: "
        f"{'Yes' if features['uses_https'] else 'No'}"
    )

    print(
        f" • Suspicious keywords: "
        f"{'Yes' if features['has_suspicious_words'] else 'No'}"
    )

    print(
        f" • URL shortener: "
        f"{'Yes' if features['has_shortener'] else 'No'}"
    )