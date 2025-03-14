# Cài đặt và load các thư viện cần thiết
library(sparklyr)
library(dplyr)
library(ggplot2)
library(tidyr)
library(gridExtra)

# Kết nối Spark
Sys.setenv(SPARK_HOME = "C:\\Users\\Admin\\Downloads\\spark-3.4.4-bin-hadoop3")
sc <- spark_connect(master = "local") 



# Đọc & Xử lý dữ liệu
covid_data <- spark_read_csv(sc, name = "covid", path = "Covid_Data.csv", header = TRUE, infer_schema = TRUE)

# Chuyển đổi cột DATE_DIED: "9999-99-99" -> 0 (sống), còn lại -> 1 (đã tử vong)
covid_data <- covid_data %>%
  mutate(DATE_DIED = if_else(DATE_DIED == "9999-99-99", 0, 1))

# Xử lý các giá trị không xác định (97, 98) thành NA
cols_to_clean <- c("INTUBED", "ICU", "PREGNANT")
covid_data <- covid_data %>%
  mutate(across(all_of(cols_to_clean), ~ if_else(. %in% c(97, 98), NA, .)))

# Chỉ giữ bệnh nhân mắc COVID (CLASIFFICATION_FINAL từ 3 đến 5)
covid_data <- covid_data %>%
  filter(CLASIFFICATION_FINAL %in% c(3, 4, 5)) %>%
  na.omit()  # Loại bỏ dòng chứa NA

# Hiển thị dữ liệu sau khi xử lý
print("Dữ liệu sau khi xử lý:")
covid_data %>% glimpse()
covid_data %>% head(10) %>% collect()

# Chia dữ liệu thành tập train/test
splits <- sdf_random_split(covid_data, training = 0.8, test = 0.2, seed = 123)
training_data <- splits$training
test_data <- splits$test

# Huấn luyện mô hình Random Forest
model_rf <- training_data %>%
  ml_random_forest_classifier(
    response = "PATIENT_TYPE",
    features = c("USMER", "AGE", "DIABETES", "OBESITY", "PNEUMONIA", "HIPERTENSION", "TOBACCO"),
    num_trees = 100,
    max_depth = 5,
    seed = 123
  )

print("Mô hình Random Forest đã huấn luyện xong!")

# Dự đoán trên tập test
predictions <- ml_predict(model_rf, test_data)

# Đánh giá mô hình
accuracy <- predictions %>%
  mutate(correct = if_else(PATIENT_TYPE == prediction, 1, 0)) %>%
  summarise(accuracy = mean(correct)) %>%
  collect()

print(paste("Độ chính xác của mô hình:", round(accuracy$accuracy * 100, 2), "%"))

# Xem 10 kết quả dự đoán đầu tiên
result_table <- predictions %>%
  select(PATIENT_TYPE, prediction) %>%
  head(10) %>%
  collect()
print("Kết quả dự đoán:")
print(result_table)

# Chuyển dữ liệu Spark về R để vẽ biểu đồ
covid_r <- covid_data %>% collect()

# Chuyển đổi dữ liệu cho biểu đồ bệnh nền
covid_melt <- covid_r %>%
  select(DIABETES, OBESITY, PNEUMONIA, HIPERTENSION, TOBACCO) %>%
  pivot_longer(cols = everything(), names_to = "Disease", values_to = "Has_Disease")


# Biểu đồ 1: Phân bố độ tuổi bệnh nhân
p1 <- ggplot(covid_r, aes(x = AGE)) +
  geom_histogram(binwidth = 5, fill = "steelblue", color = "black") +
  labs(title = "Phân bố độ tuổi bệnh nhân", x = "Tuổi", y = "Số lượng") +
  theme_minimal()

# Biểu đồ 2: Số lượng bệnh nhân mắc bệnh nền
p2 <- ggplot(covid_melt, aes(x = Disease, fill = as.factor(Has_Disease))) +
  geom_bar(position = "dodge") +
  labs(title = "Số lượng bệnh nhân mắc bệnh nền", x = "Bệnh nền", y = "Số lượng") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Biểu đồ 3: Số lượng bệnh nhân nhập ICU
p3 <- ggplot(covid_r, aes(x = as.factor(ICU), fill = as.factor(ICU))) +
  geom_bar() +
  labs(title = "Số lượng bệnh nhân nhập ICU", x = "Nhập ICU", y = "Số lượng") +
  scale_fill_manual(values = c("red", "green", "gray")) +
  theme_minimal()

# Biểu đồ 4: Mối quan hệ giữa tuổi và tình trạng nhập ICU
p4 <- ggplot(covid_r, aes(x = AGE, fill = as.factor(ICU))) +
  geom_density(alpha = 0.5) +
  labs(title = "Tuổi và tình trạng nhập ICU", x = "Tuổi", y = "Mật độ") +
  theme_minimal()

print(p1)
print(p2)
print(p3)
print(p4)

# Ngắt kết nối Spark
 spark_disconnect(sc)
 print("Kết thúc! Spark đã ngắt kết nối.")
