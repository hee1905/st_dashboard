import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import unicodedata
from sklearn.feature_extraction.text import TfidfVectorizer

# 페이지 설정
st.set_page_config(page_title="네이버 API 데이터 분석 대시보드", layout="wide")

# 데이터 경로 설정
DATA_DIR = "data"

@st.cache_data
def load_data():
    # 디버깅을 위한 로그 출력
    print(f"Current working directory: {os.getcwd()}")
    
    # 실제 서버 폴더 내 파일 목록 가져오기 (NFC 정규화)
    actual_files = {}
    if os.path.exists(DATA_DIR):
        for f in os.listdir(DATA_DIR):
            norm_f = unicodedata.normalize('NFC', f)
            actual_files[norm_f] = f
        print(f"Normalized files in {DATA_DIR}: {list(actual_files.keys())}")
    else:
        print(f"Directory {DATA_DIR} NOT FOUND")

    files = {
        "비타민D": {
            "shop": "비타민d_20260213_naver_shop.csv",
            "blog": "비타민d_20260213_naver_blog.csv",
            "trend": "비타민d_20260213_shopping_trend.csv"
        },
        "오메가3": {
            "shop": "오메가3_20260213_naver_shop.csv",
            "blog": "오메가3_20260213_naver_blog.csv",
            "trend": "오메가3_20260213_shopping_trend.csv"
        }
    }
    
    data = {}
    for kw, fset in files.items():
        data[kw] = {}
        for dtype, fname in fset.items():
            # 코드상의 파일명도 NFC 정규화
            norm_fname = unicodedata.normalize('NFC', fname)
            
            if norm_fname in actual_files:
                actual_fname = actual_files[norm_fname]
                path = os.path.join(DATA_DIR, actual_fname)
                try:
                    df = pd.read_csv(path)
                    if dtype == 'trend':
                        df['period'] = pd.to_datetime(df['period'])
                    elif dtype == 'blog':
                        df['postdate'] = pd.to_datetime(df['postdate'], format='%Y%m%d', errors='coerce')
                    elif dtype == 'shop':
                        df['lprice'] = pd.to_numeric(df['lprice'], errors='coerce')
                    data[kw][dtype] = df
                except Exception as e:
                    print(f"ERROR READING {path}: {e}")
            else:
                print(f"FILE MISSING (Normalized): {norm_fname}")
    return data

data_all = load_data()

# 사이드바 설정
st.sidebar.title("🔍 분석 설정")
selected_keywords = st.sidebar.multiselect(
    "비교할 키워드를 선택하세요",
    options=list(data_all.keys()),
    default=list(data_all.keys())
)

# 메인 화면
st.title("📊 네이버 API 데이터 분석 대시보드")
st.markdown("네이버 쇼핑, 블로그, 트렌드 데이터를 활용한 인사이트 도출")

if not selected_keywords:
    st.warning("분석할 키워드를 하나 이상 선택해주세요.")
else:
    # 탭 구성
    tab1, tab2, tab3 = st.tabs(["📈 트렌드 비교", "🛒 쇼핑 데이터 분석", "📝 블로그 키워드 분석"])

    with tab1:
        st.header("키워드별 쇼핑 검색 트렌드")
        
        # 1. 그래프: 트렌드 비교 (Line Chart)
        fig_trend = go.Figure()
        trend_found = False
        for kw in selected_keywords:
            if 'trend' in data_all[kw]:
                df_trend = data_all[kw]['trend']
                fig_trend.add_trace(go.Scatter(x=df_trend['period'], y=df_trend['ratio'], name=kw, mode='lines'))
                trend_found = True
        
        if trend_found:
            fig_trend.update_layout(title="기간별 검색 비율(ratio) 추이", xaxis_title="날짜", yaxis_title="검색 비율")
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.error("트렌드 데이터(trend CSV)를 찾을 수 없습니다. GitHub의 data 폴더에 파일이 모두 있는지 다시 확인해 주세요.")

    with tab2:
        st.header("쇼핑 시장 데이터 분석")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 2. 그래프: 가격 분포 (Box Plot)
            shop_combined = []
            for kw in selected_keywords:
                if 'shop' in data_all[kw]:
                    df_s = data_all[kw]['shop'].copy()
                    df_s['keyword'] = kw
                    shop_combined.append(df_s)
            
            if shop_combined:
                df_full_shop = pd.concat(shop_combined)
                fig_box = px.box(df_full_shop, x="keyword", y="lprice", title="키워드별 가격 분포 (Boxplot)")
                st.plotly_chart(fig_box, use_container_width=True)
            
            # 3. 그래프: 브랜드 점유율 (Bar Chart)
            st.subheader("주요 브랜드 빈도 (상위 10개)")
            brand_kw = st.selectbox("브랜드를 확인할 키워드 선택", selected_keywords, key="brand_sel")
            
            shop_data = data_all[brand_kw].get('shop')
            if shop_data is not None:
                brand_counts = shop_data['brand'].value_counts().head(10).reset_index()
                brand_counts.columns = ['brand', 'count']
                fig_brand = px.bar(brand_counts, x='brand', y='count', title=f"{brand_kw} 주요 브랜드")
                st.plotly_chart(fig_brand, use_container_width=True)
            else:
                st.error(f"{brand_kw}의 쇼핑 데이터(shop CSV)를 찾을 수 없습니다.")

        with col2:
            # 4. 그래프: 쇼핑몰별 평균 가격 (Bar Chart)
            st.subheader("쇼핑몰별 평균 가격 비교")
            mall_kw = st.selectbox("쇼핑몰을 확인할 키워드 선택", selected_keywords, key="mall_sel")
            
            mall_shop_data = data_all[mall_kw].get('shop')
            if mall_shop_data is not None:
                mall_price = mall_shop_data.groupby('mallName')['lprice'].mean().sort_values(ascending=False).head(15).reset_index()
                fig_mall = px.bar(mall_price, x='mallName', y='lprice', title=f"{mall_kw} 쇼핑몰별 평균가")
                st.plotly_chart(fig_mall, use_container_width=True)
            else:
                st.error(f"{mall_kw}의 쇼핑 데이터(shop CSV)를 찾을 수 없습니다.")

        # 표 구성
        st.divider()
        st.subheader("데이터 요약 표")
        t_col1, t_col2 = st.columns(2)
        
        with t_col1:
            st.write("📌 브랜드별 요약 (선택 키워드)")
            curr_shop_data = data_all[mall_kw].get('shop')
            if curr_shop_data is not None:
                brand_summary = curr_shop_data.groupby('brand')['lprice'].agg(['mean', 'count']).sort_values('count', ascending=False).head(10)
                st.write(brand_summary)
            
            st.write("📌 쇼핑몰별 가격 통계")
            if curr_shop_data is not None:
                mall_stats = curr_shop_data.groupby('mallName')['lprice'].agg(['min', 'max', 'mean']).head(10)
                st.write(mall_stats)
            
        with t_col2:
            st.write("📌 원본 데이터 미리보기")
            if curr_shop_data is not None:
                st.dataframe(curr_shop_data[['title', 'lprice', 'brand', 'mallName']].head(10))

    with tab3:
        st.header("블로그 게시글 키워드 분석")
        
        blog_kw = st.selectbox("블로그 분석 키워드 선택", selected_keywords, key="blog_sel")
        if 'blog' in data_all[blog_kw]:
            df_blog = data_all[blog_kw]['blog']
            
            # TF-IDF 분석
            vectorizer = TfidfVectorizer(max_features=50)
            df_blog['content'] = df_blog['title'] + " " + df_blog['description']
            tfidf_matrix = vectorizer.fit_transform(df_blog['content'].fillna(''))
            
            feature_names = vectorizer.get_feature_names_out()
            sums = tfidf_matrix.sum(axis=0)
            kw_data = []
            for col, idx in enumerate(feature_names):
                kw_data.append((idx, sums[0, col]))
            
            ranking = pd.DataFrame(kw_data, columns=['term', 'rank']).sort_values('rank', ascending=False).head(20)
            
            fig_kw = px.bar(ranking, x='term', y='rank', title=f"{blog_kw} 블로그 주요 키워드 (TF-IDF)")
            st.plotly_chart(fig_kw, use_container_width=True)
            
            st.subheader("핵심 키워드 순위표")
            st.table(ranking)
        else:
            st.error(f"{blog_kw}의 블로그 데이터(blog CSV)를 찾을 수 없습니다.")

st.sidebar.markdown("---")
st.sidebar.info("이 대시보드는 네이버 오픈 API 데이터를 기반으로 생성되었습니다.")
