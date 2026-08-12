# app.py - Coconut Farm Marketplace & Prediction System
# AI Bot: TF-IDF ML Model (No API Key, No Internet Required)

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import sqlite3
import os
from werkzeug.utils import secure_filename
from main import predict_yield, yield_recommendations, predict_disease_from_symptoms, forecast_price

# ------------------------------
# TF-IDF ML Chatbot (Local, No API)
# ------------------------------

try:
    from bot_model import load_model, get_answer
    coconut_model = load_model()
    print("Coconut TF-IDF bot loaded successfully!")
except Exception as e:
    print(f"Bot model load failed: {str(e)}")
    coconut_model = None

# ------------------------------
# REAL CNN Disease Detection (Image-based)
# ------------------------------
try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing import image
    import numpy as np
    import json
except ImportError:
    print("ERROR: tensorflow not installed. Run: pip install tensorflow")
    load_model = None

app = Flask(__name__)
app.secret_key = 'super_secret_key_12345_change_me_in_production'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# CNN Disease Model Load
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'coconut_disease_cnn.h5')
CLASSES_PATH = os.path.join(BASE_DIR, 'models', 'disease_classes.json')

disease_model = None
disease_classes = []

if os.path.exists(MODEL_PATH) and load_model:
    try:
        disease_model = load_model(MODEL_PATH)
        if os.path.exists(CLASSES_PATH):
            with open(CLASSES_PATH, 'r') as f:
                disease_classes = json.load(f)
        print(f"Disease CNN model loaded successfully! Classes: {disease_classes}")
    except Exception as e:
        print("Disease model load failed:", str(e))
else:
    print("Disease CNN model not found or tensorflow missing. Image detection will use fallback.")

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(UserMixin):
    def __init__(self, id, username, role):
        self.id = id
        self.username = username
        self.role = role

@login_manager.user_loader
def load_user(user_id):
    conn = sqlite3.connect('database.db')
    user = conn.execute('SELECT id, username, role FROM users WHERE id = ?', (user_id,)).fetchone()
    conn.close()
    if user:
        return User(user[0], user[1], user[2])
    return None

def init_db():
    conn = sqlite3.connect('database.db')
    conn.execute('''CREATE TABLE IF NOT EXISTS users 
                    (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                     username TEXT UNIQUE, 
                     password TEXT, 
                     role TEXT DEFAULT 'user')''')
    conn.execute('''CREATE TABLE IF NOT EXISTS farms 
                    (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, name TEXT, variety TEXT, age INTEGER, 
                     past_yield REAL, fertilizer REAL, irrigation TEXT, district TEXT)''')
    conn.execute('''CREATE TABLE IF NOT EXISTS listings 
                    (id INTEGER PRIMARY KEY AUTOINCREMENT, farmer_id INTEGER, title TEXT, description TEXT, 
                     quantity REAL, price REAL, district TEXT, image_path TEXT, status TEXT DEFAULT 'available')''')
    
    conn.execute('''CREATE TABLE IF NOT EXISTS harvest_plans 
                    (id INTEGER PRIMARY KEY AUTOINCREMENT, farmer_id INTEGER, farm_id INTEGER, last_harvest TEXT, 
                     trees INTEGER, tree_age INTEGER, next_harvest TEXT, estimated_production INTEGER, 
                     labor_needed INTEGER, notes TEXT)''')
    
    conn.execute('''CREATE TABLE IF NOT EXISTS farm_tasks 
                (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                 farmer_id INTEGER, 
                 task_title TEXT, 
                 task_description TEXT, 
                 due_date TEXT, 
                 frequency TEXT, 
                 status TEXT DEFAULT 'pending', 
                 created_at TEXT DEFAULT CURRENT_TIMESTAMP)''')
    
    conn.execute('''CREATE TABLE IF NOT EXISTS orders 
                    (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                     listing_id INTEGER, 
                     buyer_id INTEGER, 
                     farmer_id INTEGER,
                     quantity REAL, 
                     total_price REAL, 
                     status TEXT DEFAULT 'Paid', 
                     purchase_date TEXT DEFAULT CURRENT_TIMESTAMP)''')
    
    conn.close()

init_db()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        role = request.form.get('role', 'user')
        
        conn = sqlite3.connect('database.db')
        user = conn.execute('SELECT * FROM users WHERE username=? AND password=? AND role=?', 
                            (username, password, role)).fetchone()
        conn.close()
        
        if user:
            user_obj = User(user[0], user[1], user[3])
            login_user(user_obj)
            flash(f'Login successful as {role}!', 'success')
            if role == 'farmer':
                return redirect('/farmer_dashboard')
            elif role == 'user':
                return redirect('/user_dashboard')
            elif role == 'admin':
                return redirect('/admin_dashboard')
            else:
                return redirect('/dashboard')
        else:
            flash('Invalid username, password or role', 'danger')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        role = request.form['role']
        
        conn = sqlite3.connect('database.db')
        try:
            conn.execute('INSERT INTO users (username, password, role) VALUES (?, ?, ?)', (username, password, role))
            conn.commit()
            flash(f'Registration successful as {role}! Please login.', 'success')
            return redirect('/login')
        except sqlite3.IntegrityError:
            flash('Username already taken', 'danger')
        finally:
            conn.close()
    
    return render_template('register.html')

@app.route('/farmer_dashboard')
@login_required
def farmer_dashboard():
    if current_user.role != 'farmer':
        flash('Access denied', 'danger')
        return redirect('/dashboard')
    
    conn = sqlite3.connect('database.db')
    farms = conn.execute('SELECT * FROM farms WHERE user_id = ?', (current_user.id,)).fetchall()
    listings = conn.execute('SELECT * FROM listings WHERE farmer_id = ? AND status = "available"', (current_user.id,)).fetchall()
    conn.close()
    
    return render_template('farmer_dashboard.html', farms=farms, listings=listings, username=current_user.username)

@app.route('/user_dashboard')
@login_required
def user_dashboard():
    if current_user.role != 'user':
        flash('Access denied', 'danger')
        return redirect('/dashboard')
    
    conn = sqlite3.connect('database.db')
    listings = conn.execute('''
        SELECT l.id, l.title, l.description, l.quantity, l.price, l.district, l.image_path, l.status, u.username
        FROM listings l 
        JOIN users u ON l.farmer_id = u.id 
        WHERE l.status = 'available'
    ''').fetchall()
    conn.close()
    
    return render_template('user_dashboard.html', listings=listings, username=current_user.username)

# ------------------------------
# ADMIN SECTION
# ------------------------------

@app.route('/admin_dashboard')
@login_required
def admin_dashboard():
    if current_user.role != 'admin':
        flash('Admin access only', 'danger')
        return redirect('/farmer_dashboard')

    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row

    total_farmers = conn.execute("SELECT COUNT(*) as count FROM users WHERE role = 'farmer'").fetchone()['count']
    total_users = conn.execute("SELECT COUNT(*) as count FROM users").fetchone()['count']
    total_orders = conn.execute("SELECT COUNT(*) as count FROM orders").fetchone()['count']
    
    revenue_row = conn.execute("SELECT SUM(total_price) as total FROM orders").fetchone()
    total_revenue = revenue_row['total'] if revenue_row['total'] else 0

    active_farmers = total_farmers // 3 + 10

    recent_activities = conn.execute("""
        SELECT username, role, 'Logged in' as action, datetime('now') as time 
        FROM users 
        ORDER BY id DESC LIMIT 5
    """).fetchall()

    users_list = conn.execute('SELECT id, username, role FROM users').fetchall()
    
    farms_list = conn.execute('''
        SELECT f.id, f.name, f.variety, f.district, u.username as farmer_name
        FROM farms f
        JOIN users u ON f.user_id = u.id
    ''').fetchall()

    orders_list = conn.execute('''
        SELECT o.id, o.quantity, o.total_price, o.status, o.purchase_date,
               buyer.username as buyer_name, 
               farmer.username as farmer_name,
               l.title as product_name
        FROM orders o
        JOIN users buyer ON o.buyer_id = buyer.id
        JOIN users farmer ON o.farmer_id = farmer.id
        JOIN listings l ON o.listing_id = l.id
        ORDER BY o.purchase_date DESC
    ''').fetchall()

    conn.close()

    return render_template('admin_dashboard.html',
                           total_farmers=total_farmers,
                           active_farmers=active_farmers,
                           total_users=total_users,
                           total_orders=total_orders,
                           total_revenue=total_revenue,
                           recent_activities=recent_activities,
                           users=users_list,
                           farms=farms_list,
                           orders=orders_list)

@app.route('/admin/users')
@login_required
def admin_users():
    if current_user.role != 'admin':
        return redirect('/farmer_dashboard')
    return render_template('admin_users.html', title="Manage Users")

@app.route('/admin/farms')
@login_required
def admin_farms():
    if current_user.role != 'admin':
        return redirect('/farmer_dashboard')
    return render_template('admin_farms.html', title="All Farms")

@app.route('/admin/listings')
@login_required
def admin_listings():
    if current_user.role != 'admin':
        return redirect('/farmer_dashboard')
    return render_template('admin_listings.html', title="Marketplace Listings")

@app.route('/add_farm', methods=['POST'])
@login_required
def add_farm():
    if current_user.role != 'farmer':
        flash('Access denied', 'danger')
        return redirect('/farmer_dashboard')
    
    name = request.form['name']
    variety = request.form['variety']
    age = int(request.form['age'])
    past_yield = float(request.form.get('past_yield', 0))
    fertilizer = float(request.form.get('fertilizer', 0))
    irrigation = request.form.get('irrigation', '')
    district = request.form['district']

    conn = sqlite3.connect('database.db')
    conn.execute('INSERT INTO farms (user_id, name, variety, age, past_yield, fertilizer, irrigation, district) VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                 (current_user.id, name, variety, age, past_yield, fertilizer, irrigation, district))
    conn.commit()
    conn.close()
    
    flash('Farm added successfully!', 'success')
    return redirect('/farmer_dashboard')

@app.route('/upload_listing', methods=['POST'])
@login_required
def upload_listing():
    if current_user.role != 'farmer':
        flash('Access denied', 'danger')
        return redirect('/farmer_dashboard')
    
    title = request.form.get('title')
    description = request.form.get('description', 'No description')
    quantity = request.form.get('quantity')
    price = request.form.get('price')
    district = request.form.get('district')
    
    if not title or not quantity or not price:
        flash('Please fill all required fields', 'danger')
        return redirect('/sell_coconuts')

    image_path = None
    if 'image' in request.files:
        file = request.files['image']
        if file.filename != '':
            filename = secure_filename(file.filename)
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(image_path)

    conn = sqlite3.connect('database.db')
    conn.execute('INSERT INTO listings (farmer_id, title, description, quantity, price, district, image_path, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                 (current_user.id, title, description, float(quantity), float(price), district, image_path, 'available'))
    conn.commit()
    conn.close()
    
    flash('Coconuts uploaded successfully! It is now live in the market.', 'success')
    return redirect('/sell_coconuts')

@app.route('/farm_tasks', methods=['GET', 'POST'])
@login_required
def farm_tasks():
    if current_user.role != 'farmer':
        flash('Access denied - Only farmers can manage tasks', 'danger')
        return redirect('/farmer_dashboard')
    
    conn = sqlite3.connect('database.db')
    
    if request.method == 'POST':
        task_title = request.form.get('task_title', '').strip()
        task_desc = request.form.get('task_desc', '').strip()
        due_date = request.form.get('due_date', '')
        frequency = request.form.get('frequency', 'one-time')
        
        if not task_title or not due_date:
            flash('Task title and due date are required', 'danger')
        else:
            conn.execute('''
                INSERT INTO farm_tasks (farmer_id, task_title, task_description, due_date, frequency)
                VALUES (?, ?, ?, ?, ?)
            ''', (current_user.id, task_title, task_desc, due_date, frequency))
            conn.commit()
            flash('Task added successfully!', 'success')
    
    tasks = conn.execute('''
        SELECT id, task_title, task_description, due_date, frequency, status
        FROM farm_tasks 
        WHERE farmer_id = ? 
        ORDER BY due_date ASC
    ''', (current_user.id,)).fetchall()
    
    conn.close()
    
    return render_template('farm_tasks.html', tasks=tasks, username=current_user.username)

@app.route('/sell_coconuts')
@login_required
def sell_coconuts():
    if current_user.role != 'farmer':
        flash('Access denied', 'danger')
        return redirect('/farmer_dashboard')
    
    conn = sqlite3.connect('database.db')
    listings = conn.execute('''
        SELECT id, title, description, quantity, price, district, image_path, status 
        FROM listings 
        WHERE farmer_id = ? AND status = 'available'
    ''', (current_user.id,)).fetchall()
    conn.close()
    
    return render_template('sell_coconuts.html', listings=listings, username=current_user.username)

@app.route('/buy_coconut/<int:listing_id>', methods=['POST'])
@login_required
def buy_coconut(listing_id):
    if current_user.role != 'user':
        flash('Access denied - Only buyers can purchase', 'danger')
        return redirect('/user_dashboard')
    
    conn = sqlite3.connect('database.db')
    listing = conn.execute('SELECT status FROM listings WHERE id = ?', (listing_id,)).fetchone()
    if not listing or listing[0] != 'available':
        conn.close()
        flash('This listing is no longer available or already sold', 'danger')
        return redirect('/user_dashboard')
    
    conn.close()
    return redirect(url_for('payment', listing_id=listing_id))

@app.route('/payment/<int:listing_id>', methods=['GET', 'POST'])
@login_required
def payment(listing_id):
    if current_user.role != 'user':
        flash('Access denied', 'danger')
        return redirect('/user_dashboard')
    
    conn = sqlite3.connect('database.db')
    listing = conn.execute('''
        SELECT l.id, l.title, l.description, l.quantity, l.price, l.district, l.image_path, l.farmer_id, u.username
        FROM listings l 
        JOIN users u ON l.farmer_id = u.id 
        WHERE l.id = ? AND l.status = 'available'
    ''', (listing_id,)).fetchone()
    
    if not listing:
        conn.close()
        flash('Listing not found or already sold', 'danger')
        return redirect('/user_dashboard')
    
    if request.method == 'POST':
        try:
            buy_quantity = float(request.form['quantity'])
            if buy_quantity <= 0 or buy_quantity > listing[3]:
                flash('Invalid quantity.', 'danger')
                conn.close()
                return redirect(url_for('payment', listing_id=listing_id))
            
            total = buy_quantity * listing[4]
            
            conn.execute('''
                INSERT INTO orders (listing_id, buyer_id, farmer_id, quantity, total_price, status)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (listing[0], current_user.id, listing[7], buy_quantity, total, 'Paid'))
            
            conn.execute('UPDATE listings SET status = "sold" WHERE id = ?', (listing_id,))
            conn.commit()
            conn.close()
            
            flash(f'Payment successful! Paid Rs.{total:.2f}.', 'success')
            return redirect('/my_orders')
        except Exception as e:
            conn.close()
            flash(f'Error: {str(e)}', 'danger')
            return redirect(url_for('payment', listing_id=listing_id))
    
    conn.close()
    return render_template('payment.html', listing=listing, total=listing[4] * listing[3])

@app.route('/harvest_planning')
@login_required
def harvest_planning():
    if current_user.role != 'farmer':
        flash('Access denied', 'danger')
        return redirect('/farmer_dashboard')
    
    conn = sqlite3.connect('database.db')
    farms = conn.execute('SELECT * FROM farms WHERE user_id = ?', (current_user.id,)).fetchall()
    conn.close()
    
    return render_template('harvest_planning.html', farms=farms, username=current_user.username)

@app.route('/save_harvest_plan', methods=['POST'])
@login_required
def save_harvest_plan():
    if current_user.role != 'farmer':
        flash('Access denied', 'danger')
        return redirect('/farmer_dashboard')
    
    farm_id = request.form['farm_id']
    last_harvest = request.form['last_harvest']
    trees = int(request.form['trees'])
    tree_age = int(request.form['tree_age'])
    next_harvest = request.form['next_harvest']
    estimated_production = int(request.form['estimated_production'])
    labor_needed = int(request.form['labor_needed'])
    notes = request.form.get('notes', '')

    conn = sqlite3.connect('database.db')
    conn.execute('''
        INSERT INTO harvest_plans (farmer_id, farm_id, last_harvest, trees, tree_age, next_harvest, 
                                  estimated_production, labor_needed, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (current_user.id, farm_id, last_harvest, trees, tree_age, next_harvest, 
          estimated_production, labor_needed, notes))
    conn.commit()
    conn.close()
    
    flash('Harvest Plan saved successfully!', 'success')
    return redirect(f'/farm/{farm_id}')

@app.route('/predict_yield_price', methods=['GET', 'POST'])
@login_required
def predict_yield_price():
    if current_user.role != 'farmer':
        flash('Access denied', 'danger')
        return redirect('/farmer_dashboard')

    prediction = None
    recommendations = None
    price_forecast = None
    forecast_error = None

    if request.method == 'POST':
        variety = request.form.get('variety', 'West Coast Tall')
        location = request.form.get('location', 'Coimbatore')
        soil_type = request.form.get('soil_type', 'Loamy')
        weather = request.form.get('weather', 'Normal')
        tree_age = int(request.form.get('tree_age', 8))
        area_ha = float(request.form.get('area', 5))

        inputs = {
            'crop_year': 2025, 'district_name': location, 'area': area_ha * 10000,
            'season': 'Kharif', 'crop': variety
        }

        prediction = predict_yield(inputs)
        if prediction is not None:
            prediction = round(prediction * area_ha, 2)
            recommendations = yield_recommendations(prediction)
        else:
            prediction = "Error"

        periods = int(request.form.get('periods', 30))
        price_forecast = forecast_price(location, periods)
        if isinstance(price_forecast, dict) and 'error' in price_forecast:
            forecast_error = price_forecast['error']

    return render_template('predict_yield_price.html', prediction=prediction, recommendations=recommendations,
                          price_forecast=price_forecast, forecast_error=forecast_error)

@app.route('/predict_disease', methods=['GET', 'POST'])
@login_required
def predict_disease():
    if current_user.role != 'farmer':
        flash('Access denied', 'danger')
        return redirect('/farmer_dashboard')

    disease = None
    disease_solution = None
    uploaded_image = None
    detection_method = "No detection"
    confidence = 0.0

    if request.method == 'POST':
        symptoms = request.form.get('symptoms', '').strip().lower()
        if symptoms:
            disease, disease_solution = predict_disease_from_symptoms(symptoms)
            detection_method = "Based on symptoms"
            confidence = 75.0

        if 'image' in request.files:
            file = request.files['image']
            if file.filename != '':
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                uploaded_image = '/static/uploads/' + filename

                if disease_model and disease_classes:
                    try:
                        img = image.load_img(file_path, target_size=(224, 224))
                        img_array = image.img_to_array(img)
                        img_array = np.expand_dims(img_array, axis=0)
                        img_array = img_array / 255.0

                        preds = disease_model.predict(img_array)
                        print(f"DEBUG Raw Predictions: {preds[0]}")
                        
                        predicted_idx = np.argmax(preds[0])
                        raw_confidence = float(preds[0][predicted_idx]) * 100
                        print(f"DEBUG Predicted Class Index: {predicted_idx}, Confidence: {raw_confidence}")

                        predicted_disease_name = disease_classes[predicted_idx]

                        if "healthy" in predicted_disease_name.lower() or "good" in predicted_disease_name.lower():
                            disease = "Healthy Leaf (No Disease)"
                            confidence = 0.0
                            disease_solution = "Your tree appears healthy!"
                            detection_method = "CNN Image Detection (Healthy)"
                        else:
                            if raw_confidence > 40:
                                disease = predicted_disease_name
                                confidence = raw_confidence
                                detection_method = "CNN Image Detection"
                                treatments = {
                                    'Bud Rot': "1. Remove affected parts.\n2. Apply Bordeaux paste.",
                                    'Stem Bleeding': "1. Clean wound.\n2. Apply Coal tar.",
                                    'CCI_Leaflets': "1. Check for root rot.\n2. Improve drainage.",
                                }
                                disease_solution = treatments.get(disease, "Consult local agriculture officer.")
                            else:
                                disease = "Uncertain"
                                confidence = 20.0
                                disease_solution = "Model confidence low. Try clearer image."
                            
                    except Exception as e:
                        disease = "Prediction Error"
                        disease_solution = str(e)
                else:
                    disease = "Model Error"
                    disease_solution = "AI Model not loaded."

    return render_template('predict_disease.html',
                          disease=disease,
                          disease_solution=disease_solution,
                          uploaded_image=uploaded_image,
                          detection_method=detection_method,
                          confidence=round(confidence, 2))

# ------------------------------
# AI CHAT - TF-IDF ML Bot (No API Key)
# ------------------------------
@app.route('/ai_chat', methods=['POST'])
@login_required
def ai_chat():
    user_message = request.json.get('message', '').strip()
    if not user_message:
        return jsonify({'reply': 'Please ask something!'})
    
    if not coconut_model:
        return jsonify({'reply': 'AI bot not initialized. Please run: python bot_model.py first!'})
    
    try:
        reply = get_answer(user_message, coconut_model)
    except Exception as e:
        reply = f"Error: {str(e)}"
    
    return jsonify({'reply': reply})

@app.route('/guide')
def guide():
    return render_template('guide.html')

# ------------------------------
# ORDER HISTORY ROUTES
# ------------------------------

@app.route('/my_orders')
@login_required
def my_orders():
    if current_user.role != 'user':
        return redirect('/farmer_dashboard')
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    orders = conn.execute('''
        SELECT o.id, o.quantity, o.total_price, o.status, o.purchase_date,
               l.title, l.image_path, l.district, u.username as farmer_name
        FROM orders o JOIN listings l ON o.listing_id = l.id
        JOIN users u ON o.farmer_id = u.id
        WHERE o.buyer_id = ? ORDER BY o.purchase_date DESC
    ''', (current_user.id,)).fetchall()
    conn.close()
    return render_template('my_orders.html', orders=orders, username=current_user.username)


@app.route('/my_sales')
@login_required
def my_sales():
    if current_user.role != 'farmer':
        return redirect('/user_dashboard')
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    sales = conn.execute('''
        SELECT o.id, o.quantity, o.total_price, o.status, o.purchase_date,
               l.title, l.image_path, u.username as buyer_name
        FROM orders o JOIN listings l ON o.listing_id = l.id
        JOIN users u ON o.buyer_id = u.id
        WHERE o.farmer_id = ? ORDER BY o.purchase_date DESC
    ''', (current_user.id,)).fetchall()
    conn.close()
    return render_template('my_sales.html', sales=sales, username=current_user.username)

@app.route('/farm/<int:farm_id>')
@login_required
def farm_detail(farm_id):
    if current_user.role != 'farmer':
        flash('Access denied', 'danger')
        return redirect('/login')

    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    
    farm = conn.execute('SELECT * FROM farms WHERE id = ? AND user_id = ?', (farm_id, current_user.id)).fetchone()
    
    if not farm:
        conn.close()
        flash("Farm not found or access denied", "danger")
        return redirect('/farmer_dashboard')

    tasks = conn.execute('''
        SELECT id, task_title, task_description, due_date, status 
        FROM farm_tasks 
        WHERE farmer_id = ? 
        ORDER BY due_date ASC LIMIT 5
    ''', (current_user.id,)).fetchall()

    harvest_plan = conn.execute('''
        SELECT * FROM harvest_plans 
        WHERE farm_id = ? 
        ORDER BY id DESC LIMIT 1
    ''', (farm_id,)).fetchone()

    conn.close()
    
    return render_template('farm_detail.html', 
                           farm=farm, 
                           tasks=tasks, 
                           plan=harvest_plan, 
                           username=current_user.username)

@app.route('/logout', methods=['GET', 'POST'])
@login_required
def logout():
    logout_user()
    flash('Logged out successfully', 'info')
    return redirect('/')

if __name__ == '__main__':
    app.run(debug=True, port=5000)
    
    
    